import argparse
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from common_utils.image import resolve_structdiff_image_path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive SAM2 mask editor for StructDiff.")
    parser.add_argument("--image_name", type=str, default=None, help="Initial image name from data/ or explicit image path.")
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=None,
        help="Local SAM2.1 checkpoint path. If omitted, StructDiff looks for sam2/checkpoints/sam2.1_hiera_large.pt.",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Vendored SAM2 config path used with --sam2_checkpoint.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override, for example cuda or cpu.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the local Gradio server.")
    parser.add_argument("--port", type=int, default=7861, help="Port for the local Gradio server.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share mode.")
    return parser.parse_args()


def structdiff_root():
    return Path(__file__).resolve().parent


def data_root():
    return structdiff_root() / "data"


def repo_image_directory():
    return data_root() / "images"


def vendored_sam2_root():
    return structdiff_root() / "sam2"


def default_sam2_checkpoint_path():
    return vendored_sam2_root() / "checkpoints" / "sam2.1_hiera_large.pt"


def ensure_localhost_bypasses_proxy(host):
    # Gradio's startup probe can fail if localhost traffic is routed through an HTTP proxy.
    no_proxy_hosts = {"127.0.0.1", "localhost", "::1"}
    if host:
        no_proxy_hosts.add(host)

    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = [entry.strip() for entry in existing.split(",") if entry.strip()]
    for value in sorted(no_proxy_hosts):
        if value not in entries:
            entries.append(value)

    merged = ",".join(entries)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def is_readable_image(path):
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def list_available_images():
    options = []
    directory = repo_image_directory()
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if is_readable_image(path):
                options.append((path.name, path.name))
    return options


def default_mask_path(image_name):
    return data_root() / "masks" / "default" / f"{Path(image_name).stem}_mask.png"


def variant_mask_path(image_name, variant_name):
    image_stem = Path(image_name).stem
    normalized = variant_name.strip()
    if not normalized:
        raise ValueError("Variant name must not be empty.")
    if normalized.endswith(".png"):
        filename = normalized
    else:
        if normalized.startswith(f"{image_stem}_"):
            filename = f"{normalized}.png"
        else:
            filename = f"{image_stem}_{normalized}.png"
    return data_root() / "masks" / "variants" / image_stem / filename


def load_mask_as_bool(mask_path, expected_shape=None):
    if not mask_path.exists():
        return None, None

    try:
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask, dtype=np.uint8) > 127
    except Exception as exc:
        return None, f"Ignored default mask because it could not be read: {mask_path} ({exc})"

    if expected_shape is not None and tuple(mask_array.shape[:2]) != tuple(expected_shape[:2]):
        return (
            None,
            "Ignored default mask because its size does not match the image: "
            f"{mask_path} ({mask_array.shape[1]}x{mask_array.shape[0]} vs "
            f"{expected_shape[1]}x{expected_shape[0]})",
        )

    return mask_array, None


def save_mask(mask, output_path):
    # Save as a 3-channel PNG so the result is easy to inspect in standard image viewers.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_rgb = np.repeat(mask_uint8[:, :, None], 3, axis=2)
    Image.fromarray(mask_rgb).save(output_path)
    return output_path


class MaskTool:
    """Interactive SAM2-based foreground mask editor for StructDiff."""

    def __init__(self, args):
        self.args = args
        self.device = self._resolve_device(args.device)
        self.predictor = self._build_predictor()
        self.available_images = list_available_images()
        self.available_image_values = [value for _, value in self.available_images]

        self.image_name = None
        self.image_path = None
        self.image_source = None
        self.image_rgb = None
        self.current_mask = None
        self.points = []
        self.labels = []
        self.last_scores = None

    def _resolve_device(self, device_arg):
        if device_arg:
            return torch.device(device_arg)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _import_sam2(self):
        repo_dir = vendored_sam2_root().resolve()
        if not repo_dir.exists():
            raise ImportError(
                f"Vendored SAM2 directory was not found at {repo_dir}. "
                "StructDiff expects the local SAM2 runtime to live under ./sam2."
            )
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "Could not import the vendored SAM2 runtime from ./sam2.\n"
                "Install the Python dependencies from requirements_mask.txt in a SAM2-compatible environment.\n"
                "Using a separate SAM2-compatible environment is recommended because official SAM2 requires newer "
                "torch/torchvision versions than this StructDiff training release."
            ) from exc
        return build_sam2, SAM2ImagePredictor

    def _resolve_checkpoint_path(self):
        if self.args.sam2_checkpoint is not None:
            checkpoint_path = Path(self.args.sam2_checkpoint).expanduser().resolve()
        else:
            checkpoint_path = default_sam2_checkpoint_path().resolve()

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "SAM2 checkpoint not found. "
                f"Expected {checkpoint_path}. "
                "Place a SAM2.1 checkpoint there or pass --sam2_checkpoint explicitly."
            )
        return checkpoint_path

    def _build_predictor(self):
        build_sam2, SAM2ImagePredictor = self._import_sam2()
        checkpoint_path = self._resolve_checkpoint_path()
        model = build_sam2(self.args.sam2_config, str(checkpoint_path), device=str(self.device))
        predictor = SAM2ImagePredictor(model)
        return predictor

    @contextmanager
    def predictor_context(self):
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    yield
            else:
                with nullcontext():
                    yield

    def reset_prompts(self):
        self.points = []
        self.labels = []
        self.last_scores = None

    def initial_repo_choice(self):
        if not self.args.image_name:
            return None

        candidate = Path(self.args.image_name.strip()).name
        return candidate if candidate in self.available_image_values else None

    def current_paths_text(self):
        if self.image_name is None:
            return "No image loaded yet."

        variant_dir = data_root() / "masks" / "variants" / Path(self.image_name).stem
        source_text = self.image_source if self.image_source is not None else "external path"
        return "\n".join(
            [
                f"Image source: {source_text}",
                f"Loaded image path: {self.image_path}",
                f"Save Current Mask as Default -> {default_mask_path(self.image_name)}",
                f"Save Current Mask as Variant -> {variant_dir}/<variant_name>.png",
            ]
        )

    def empty_outputs(self, status_message):
        return None, None, status_message, self.current_paths_text()

    def resolve_image_path(self, explicit_path, image_name, uploaded_path):
        # Uploaded or explicit local files take precedence over repo images.
        if uploaded_path:
            candidate = Path(uploaded_path).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(f"Uploaded image path does not exist: {candidate}")
            return candidate.resolve(), candidate.name, "uploaded file"

        chosen = explicit_path.strip() if explicit_path else ""
        if chosen:
            candidate = Path(chosen).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(f"Image path does not exist: {candidate}")
            return candidate.resolve(), candidate.name, "explicit path"

        if not image_name:
            raise ValueError("Please choose an image from the dropdown or provide an explicit path.")

        resolved = resolve_structdiff_image_path(image_name)
        repo_value = f"data/images/{Path(resolved).name}"
        return Path(resolved), Path(resolved).name, repo_value

    def render_overlay(self):
        if self.image_rgb is None:
            return None

        overlay = self.image_rgb.copy()
        if self.current_mask is not None:
            mask_color = np.zeros_like(overlay)
            mask_color[:, :, 1] = 255
            overlay = np.where(
                self.current_mask[:, :, None],
                (0.55 * overlay + 0.45 * mask_color).astype(np.uint8),
                overlay,
            )

        for (x, y), label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(overlay, (x, y), radius=6, color=color, thickness=-1)
            cv2.circle(overlay, (x, y), radius=10, color=(255, 255, 255), thickness=1)

        return overlay

    def render_mask_preview(self):
        if self.image_rgb is None:
            return None
        if self.current_mask is None:
            return np.zeros_like(self.image_rgb)
        mask_uint8 = (self.current_mask.astype(np.uint8) * 255)
        return np.repeat(mask_uint8[:, :, None], 3, axis=2)

    def status_text(self, extra_message=None):
        if self.image_rgb is None:
            return "Load an image to begin."

        parts = [f"Image: {self.image_path}"]
        if self.last_scores is not None:
            rounded = ", ".join(f"{score:.3f}" for score in self.last_scores)
            parts.append(f"SAM2 mask scores: {rounded}")
        if extra_message:
            parts.append(extra_message)
        return "\n".join(parts)

    def load_image(self, image_name, explicit_path, uploaded_path):
        try:
            resolved_path, resolved_name, resolved_source = self.resolve_image_path(explicit_path, image_name, uploaded_path)
            image_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {resolved_path}")

            self.image_name = resolved_name
            self.image_path = str(resolved_path)
            self.image_source = resolved_source
            self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.reset_prompts()

            existing_mask, mask_message = load_mask_as_bool(
                default_mask_path(self.image_name),
                expected_shape=self.image_rgb.shape[:2],
            )
            self.current_mask = existing_mask

            with self.predictor_context():
                self.predictor.set_image(self.image_rgb)

            if existing_mask is not None:
                message = f"Loaded existing default mask: {default_mask_path(self.image_name)}"
            elif mask_message is not None:
                message = mask_message
            else:
                message = "No existing default mask found. Start clicking positive/negative points."

            return self.render_overlay(), self.render_mask_preview(), self.status_text(message), self.current_paths_text()
        except Exception as exc:
            return self.empty_outputs(f"Failed to load image: {exc}")

    def predict_mask(self):
        if self.image_rgb is None or not self.points:
            return

        point_coords = np.array(self.points, dtype=np.float32)
        point_labels = np.array(self.labels, dtype=np.int32)

        with self.predictor_context():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        # SAM2 returns multiple candidates; keep the highest-scoring foreground mask.
        best_index = int(np.argmax(scores))
        self.current_mask = masks[best_index].astype(bool)
        self.last_scores = scores.tolist()

    def add_point(self, click_mode, evt: gr.SelectData):
        if self.image_rgb is None:
            return self.empty_outputs("Load an image before adding prompt points.")

        x, y = evt.index
        x = int(np.clip(x, 0, self.image_rgb.shape[1] - 1))
        y = int(np.clip(y, 0, self.image_rgb.shape[0] - 1))

        self.points.append((x, y))
        self.labels.append(1 if click_mode == "Positive" else 0)
        self.predict_mask()

        return (
            self.render_overlay(),
            self.render_mask_preview(),
            self.status_text(f"Added {'positive' if click_mode == 'Positive' else 'negative'} point at ({x}, {y})."),
            self.current_paths_text(),
        )

    def undo_point(self):
        if not self.points:
            return (
                self.render_overlay(),
                self.render_mask_preview(),
                self.status_text("No prompt points to undo."),
                self.current_paths_text(),
            )

        self.points.pop()
        self.labels.pop()
        self.current_mask = None
        self.last_scores = None
        if self.points:
            self.predict_mask()

        return (
            self.render_overlay(),
            self.render_mask_preview(),
            self.status_text("Removed the last prompt point."),
            self.current_paths_text(),
        )

    def clear_points(self):
        self.reset_prompts()
        self.current_mask = None
        return (
            self.render_overlay(),
            self.render_mask_preview(),
            self.status_text("Cleared prompt points and prediction."),
            self.current_paths_text(),
        )

    def save_default_mask(self):
        if self.image_name is None or self.current_mask is None:
            return self.status_text("Nothing to save yet. Load an image and generate a mask first."), self.current_paths_text()

        output_path = save_mask(self.current_mask, default_mask_path(self.image_name))
        return self.status_text(f"Saved current mask as the default mask: {output_path}"), self.current_paths_text()

    def save_variant_mask(self, variant_name):
        if self.image_name is None or self.current_mask is None:
            return self.status_text("Nothing to save yet. Load an image and generate a mask first."), self.current_paths_text()

        try:
            output_path = save_mask(self.current_mask, variant_mask_path(self.image_name, variant_name))
        except Exception as exc:
            return self.status_text(f"Failed to save variant mask: {exc}"), self.current_paths_text()

        return self.status_text(f"Saved current mask as a variant mask: {output_path}"), self.current_paths_text()

    def build_demo(self):
        initial_choice = self.initial_repo_choice()

        with gr.Blocks(title="StructDiff Mask Tool") as demo:
            with gr.Row():
                image_dropdown = gr.Dropdown(
                    choices=self.available_images,
                    value=initial_choice,
                    label="Repo Image",
                    allow_custom_value=False,
                )
                explicit_path = gr.Textbox(
                    value=self.args.image_name if self.args.image_name and initial_choice is None else "",
                    label="Explicit Image Path",
                    placeholder="/path/to/image.png",
                )
                upload_file = gr.File(
                    label="Upload Local Image",
                    file_types=["image"],
                    type="filepath",
                )
                load_button = gr.Button("Load Image", variant="primary")

            with gr.Row():
                click_mode = gr.Radio(
                    choices=["Positive", "Negative"],
                    value="Positive",
                    label="Click Mode",
                )
                undo_button = gr.Button("Undo Point")
                clear_button = gr.Button("Clear Points")

            with gr.Row():
                overlay_image = gr.Image(
                    label="Interactive Overlay (click only)",
                    type="numpy",
                    interactive=True,
                    sources=[],
                    buttons=["fullscreen"],
                )
                mask_preview = gr.Image(
                    label="Current Mask",
                    type="numpy",
                    interactive=False,
                    buttons=["download", "fullscreen"],
                )

            status_box = gr.Textbox(label="Status", lines=5, interactive=False)
            paths_box = gr.Textbox(label="Loaded Image / Save Paths", lines=6, interactive=False)

            with gr.Row():
                save_default_button = gr.Button("Save Current Mask as Default", variant="primary")
                variant_name = gr.Textbox(
                    label="Variant Name",
                    placeholder="mask_change0 or face_3_mask_change0.png",
                )
                save_variant_button = gr.Button("Save Current Mask as Variant")

            load_button.click(
                self.load_image,
                inputs=[image_dropdown, explicit_path, upload_file],
                outputs=[overlay_image, mask_preview, status_box, paths_box],
            )
            overlay_image.select(
                self.add_point,
                inputs=[click_mode],
                outputs=[overlay_image, mask_preview, status_box, paths_box],
            )
            undo_button.click(
                self.undo_point,
                outputs=[overlay_image, mask_preview, status_box, paths_box],
            )
            clear_button.click(
                self.clear_points,
                outputs=[overlay_image, mask_preview, status_box, paths_box],
            )
            save_default_button.click(
                self.save_default_mask,
                outputs=[status_box, paths_box],
            )
            save_variant_button.click(
                self.save_variant_mask,
                inputs=[variant_name],
                outputs=[status_box, paths_box],
            )

            demo.load(
                self.load_image,
                inputs=[image_dropdown, explicit_path, upload_file],
                outputs=[overlay_image, mask_preview, status_box, paths_box],
            )

        return demo


def main():
    args = parse_args()
    ensure_localhost_bypasses_proxy(args.host)
    tool = MaskTool(args)
    demo = tool.build_demo()
    demo.queue(api_open=False).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
