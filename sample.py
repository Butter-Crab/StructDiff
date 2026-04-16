import os
from pathlib import Path

import torch
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from common_utils.common import two_tuple
from common_utils.image import (
    load_structdiff_image,
    load_structdiff_image_and_mask,
    load_structdiff_mask_variant,
)
from config import Config, log_config, parse_cmdline_args_to_config
from diffusion.diffusion import Diffusion
from diffusion.diffusion_utils import save_diffusion_sample
from diffusion.image_sampling import (
    build_outpainting_mask,
    build_region_mask,
    sample_with_outpainting,
    sample_with_reference_guidance,
    sample_with_text_guidance,
)
from models.arfnet import ARFNet
from models.positional_controls import (
    copy_shift_grid,
    foreground_mask_from_grid,
    scale_grid,
    shift_grid,
)
from models.positional_encoding import PositionalEncoding3D


def get_model_path(image_name, version_name):
    return os.path.join("lightning_logs", image_name, version_name, "checkpoints", "last.ckpt")


def _slug(value):
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe[:80] if safe else "sample"


def create_sample_directory(cfg):
    image_tag = Path(cfg.image_name).stem
    parts = [cfg.output_dir, image_tag, cfg.run_name or "default_run", cfg.sample_mode]
    if cfg.sample_mode == "control":
        parts.append("identity" if cfg.control_action == "none" else cfg.control_action)
    elif cfg.sample_mode == "text":
        parts.append(_slug(cfg.text_input))
    sample_directory = os.path.join(*parts)
    os.makedirs(sample_directory, exist_ok=True)
    print(f"Sample directory: {sample_directory}")
    return sample_directory


def get_default_sample_size(image):
    _, _, height, width = image.shape
    return height, width


def resize_tensor_image(image, size, mode="bilinear"):
    size = two_tuple(size)
    kwargs = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(image, size=size, mode=mode, **kwargs)


def normalize_for_diffusion(image):
    return image * 2 - 1


def save_mask_preview(mask, output_path):
    mask_rgb = mask[:, :1].repeat(1, 3, 1, 1)
    save_diffusion_sample(mask_rgb * 2 - 1, output_path)


def build_head_grid(cfg, sample_size, mask):
    if not cfg.use_positional_encoding:
        return None

    head_position_encode = PositionalEncoding3D()
    dummy = torch.zeros(1, 3, sample_size[0], sample_size[1], dtype=torch.float32)
    return head_position_encode(dummy, mask, fg_bg=True)


def apply_control_action(head_grid, cfg):
    if cfg.control_action == "none":
        return head_grid
    if cfg.control_action == "shift":
        return shift_grid(head_grid, x_shift=cfg.shift_x, y_shift=cfg.shift_y)
    if cfg.control_action == "copy_shift":
        return copy_shift_grid(head_grid, x_shift=cfg.shift_x, y_shift=cfg.shift_y)
    if cfg.control_action == "scale":
        return scale_grid(head_grid, scale_factor=cfg.scale_factor)
    raise ValueError(f"Unsupported control action: {cfg.control_action}")


def load_sampling_inputs(cfg):
    image, default_mask, image_path, mask_path, used_full_foreground_mask = load_structdiff_image_and_mask(
        cfg.image_name,
        resize_max_side=cfg.resize_max_side,
    )
    sample_size = two_tuple(cfg.sample_size) if cfg.sample_size is not None else get_default_sample_size(image)

    resized_default_mask = resize_tensor_image(default_mask, sample_size, mode="nearest")

    active_mask = resized_default_mask
    active_mask_path = mask_path
    if cfg.mask_variant is not None:
        variant_mask, variant_mask_path = load_structdiff_mask_variant(
            cfg.image_name,
            cfg.mask_variant,
            resize_max_side=cfg.resize_max_side,
        )
        active_mask = resize_tensor_image(variant_mask, sample_size, mode="nearest")
        active_mask_path = variant_mask_path
        used_full_foreground_mask = False

    return {
        "image_path": image_path,
        "sample_size": sample_size,
        "active_mask": active_mask,
        "active_mask_path": active_mask_path,
        "used_full_foreground_mask": used_full_foreground_mask,
    }


def print_sampling_inputs(info):
    print(f"Sampling image: {info['image_path']}")
    if info["active_mask_path"] is not None:
        print(f"Foreground mask: {info['active_mask_path']}")
    elif info["used_full_foreground_mask"]:
        print("Foreground mask: <not found in data/masks/default, using full-image foreground>")


def load_model(cfg, device, head_grid=None):
    if cfg.run_name is None:
        raise ValueError("Sampling requires --run_name so the checkpoint can be located.")

    checkpoint_path = get_model_path(cfg.image_name, cfg.run_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_pe = checkpoint.get("hyper_parameters", {}).get("uses_positional_encoding")
    if expected_pe is True and head_grid is None:
        raise ValueError(
            "This checkpoint was trained with positional encoding. Re-run sampling with --use_positional_encoding."
        )
    if expected_pe is False and head_grid is not None:
        raise ValueError(
            "This checkpoint was trained without positional encoding. Remove --use_positional_encoding."
        )

    return Diffusion.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model=ARFNet(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth),
        timesteps=cfg.diffusion_timesteps,
        head_grid=head_grid,
        training_target="x0",
        noise_schedule="cosine",
    ).to(device).eval()


def resolve_guidance_window(cfg, model):
    start_step = model.num_timesteps if cfg.guidance_start_step is None else cfg.guidance_start_step
    stop_step = cfg.guidance_stop_step
    if stop_step < 0:
        raise ValueError("guidance_stop_step must be non-negative.")
    if start_step <= stop_step:
        raise ValueError("guidance_start_step must be greater than guidance_stop_step.")
    return start_step, stop_step


def load_reference_image(reference_name_or_path, target_size=None, resize_max_side=None):
    reference_image, reference_path = load_structdiff_image(
        reference_name_or_path,
        resize_max_side=resize_max_side,
    )
    if target_size is not None:
        reference_image = resize_tensor_image(reference_image, target_size, mode="bilinear")
    return normalize_for_diffusion(reference_image), reference_path


def sample_batches(cfg, sample_fn):
    samples = []
    for start in tqdm(range(0, cfg.sample_count, cfg.batch_size)):
        current_batch = min(cfg.batch_size, cfg.sample_count - start)
        samples.append(sample_fn(current_batch))
    return torch.cat(samples, dim=0)


def generate_samples(cfg):
    sample_directory = create_sample_directory(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = load_sampling_inputs(cfg)
    print_sampling_inputs(inputs)
    save_mask_preview(inputs["active_mask"], os.path.join(sample_directory, "conditioning_mask.png"))

    head_grid = build_head_grid(cfg, inputs["sample_size"], inputs["active_mask"])
    if cfg.sample_mode == "control":
        if not cfg.use_positional_encoding:
            raise ValueError("Control mode requires --use_positional_encoding.")
        head_grid = apply_control_action(head_grid, cfg)
        edited_foreground_mask = foreground_mask_from_grid(head_grid)
        save_mask_preview(edited_foreground_mask, os.path.join(sample_directory, "control_mask.png"))

    model = load_model(cfg, device=device, head_grid=head_grid)

    if cfg.sample_mode == "diverse" or cfg.sample_mode == "control":
        samples = sample_batches(
            cfg,
            lambda batch_size: model.sample(image_size=inputs["sample_size"], batch_size=batch_size),
        )

    elif cfg.sample_mode == "text":
        try:
            from text2live_util.clip_extractor import ClipExtractor
            from text2live_util.util import get_augmentations_template
        except ImportError as exc:
            raise ImportError(
                "Text-guided sampling requires the CLIP dependency. "
                "Install the package listed in requirements.txt before using --sample_mode text."
            ) from exc

        clip_cfg = {
            "clip_model_name": "ViT-B/32",
            "clip_affine_transform_fill": True,
            "n_aug": 16,
        }
        clip_model = ClipExtractor(clip_cfg)
        text_embeddings = clip_model.get_text_embedding(cfg.text_input, template=get_augmentations_template("lr"))
        region_mask = build_region_mask(
            cfg.guidance_region,
            (inputs["active_mask"][:, :1] > 0.5).float(),
            batch_size=1,
            image_channels=3,
        )
        start_step, stop_step = resolve_guidance_window(cfg, model)
        samples = sample_batches(
            cfg,
            lambda batch_size: sample_with_text_guidance(
                model,
                image_size=inputs["sample_size"],
                batch_size=batch_size,
                clip_model=clip_model,
                text_embeddings=text_embeddings,
                region_mask=region_mask,
                clip_strength=cfg.text_guidance_strength,
                guidance_sub_iters=cfg.guidance_sub_iters,
                start_step=start_step,
                stop_step=stop_step,
            ),
        )

    elif cfg.sample_mode == "reference":
        reference_name = cfg.reference_image or cfg.image_name
        reference_image, reference_path = load_reference_image(
            reference_name,
            target_size=inputs["sample_size"],
            resize_max_side=cfg.resize_max_side,
        )
        print(f"Reference image: {reference_path}")
        start_step, stop_step = resolve_guidance_window(cfg, model)
        samples = sample_batches(
            cfg,
            lambda batch_size: sample_with_reference_guidance(
                model,
                image_size=inputs["sample_size"],
                batch_size=batch_size,
                reference_image=reference_image.to(device),
                start_step=start_step,
                stop_step=stop_step,
                downsample_factor=cfg.reference_downsample_factor,
            ),
        )

    elif cfg.sample_mode == "outpaint":
        reference_name = cfg.reference_image or cfg.image_name
        raw_reference_image, reference_path = load_structdiff_image(
            reference_name,
            resize_max_side=cfg.resize_max_side,
        )
        reference_size = get_default_sample_size(raw_reference_image)
        reference_image = normalize_for_diffusion(
            resize_tensor_image(raw_reference_image, reference_size, mode="bilinear")
        )
        print(f"Reference image: {reference_path}")
        outpainting_mask = build_outpainting_mask(
            reference_image,
            canvas_size=inputs["sample_size"],
            offset=cfg.outpaint_offset,
            batch_size=1,
            device=device,
        )
        save_mask_preview(outpainting_mask, os.path.join(sample_directory, "outpainting_mask.png"))
        start_step, stop_step = resolve_guidance_window(cfg, model)
        samples = sample_batches(
            cfg,
            lambda batch_size: sample_with_outpainting(
                model,
                image_size=inputs["sample_size"],
                batch_size=batch_size,
                reference_image=reference_image.to(device),
                outpainting_mask=outpainting_mask,
                start_step=start_step,
                stop_step=stop_step,
            ),
        )

    else:
        raise ValueError(f"Unsupported sample_mode: {cfg.sample_mode}")

    save_diffusion_sample(samples, os.path.join(sample_directory, "sample.png"))


def main():
    cfg = Config()
    cfg = parse_cmdline_args_to_config(cfg)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.available_gpus

    log_config(cfg, context="sample")
    generate_samples(cfg)


if __name__ == "__main__":
    main()
