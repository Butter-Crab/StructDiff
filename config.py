import argparse


class Config:
    project_name = "StructDiff"

    available_gpus = "0"
    run_name = None

    task = "image"
    diffusion_timesteps = 50

    network_filters = 64
    network_depth = 16

    image_name = "balloons.png"
    initial_lr = 0.0002
    resize_max_side = 256

    output_dir = "outputs"
    sample_mode = "diverse"
    sample_count = 1
    batch_size = 8
    sample_size = None

    use_positional_encoding = False
    mask_variant = None

    control_action = "none"
    shift_x = 0
    shift_y = 0
    scale_factor = 1.0

    text_input = "forest"
    text_guidance_strength = 0.6
    guidance_sub_iters = 1
    guidance_region = "foreground"
    guidance_start_step = None
    guidance_stop_step = 2

    reference_image = None
    reference_downsample_factor = 4.0

    outpaint_offset = (0, 0)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def log_config(cfg, context=None):
    def append(parts, label, value):
        if value is None:
            return
        parts.append(f"{label}={value}")

    parts = [
        f"task={cfg.task}",
        f"image={cfg.image_name}",
        f"run={cfg.run_name or 'default'}",
        f"pe={'on' if cfg.use_positional_encoding else 'off'}",
    ]
    append(parts, "resize_max_side", cfg.resize_max_side if cfg.resize_max_side else "off")

    if context == "train":
        append(parts, "timesteps", cfg.diffusion_timesteps)
        append(parts, "lr", cfg.initial_lr)
        append(parts, "depth", cfg.network_depth)
        append(parts, "filters", cfg.network_filters)
        print("Training config: " + " | ".join(parts))
        return

    if context == "sample":
        append(parts, "mode", cfg.sample_mode)
        append(parts, "count", cfg.sample_count)
        append(parts, "batch", cfg.batch_size)
        append(parts, "size", cfg.sample_size)
        append(parts, "mask_variant", cfg.mask_variant)

        if cfg.sample_mode == "control":
            append(parts, "control", "identity" if cfg.control_action == "none" else cfg.control_action)
            if cfg.control_action in {"shift", "copy_shift"}:
                append(parts, "shift", (cfg.shift_x, cfg.shift_y))
            elif cfg.control_action == "scale":
                append(parts, "scale", cfg.scale_factor)
        elif cfg.sample_mode == "text":
            append(parts, "text", cfg.text_input)
            append(parts, "guidance", cfg.text_guidance_strength)
            append(parts, "region", cfg.guidance_region)
        elif cfg.sample_mode == "reference":
            append(parts, "reference", cfg.reference_image or cfg.image_name)
            append(parts, "downsample", cfg.reference_downsample_factor)
        elif cfg.sample_mode == "outpaint":
            append(parts, "reference", cfg.reference_image or cfg.image_name)
            append(parts, "offset", cfg.outpaint_offset)

        print("Sampling config: " + " | ".join(parts))
        return

    print("Config: " + " | ".join(parts))


def _tuple_of_ints(value):
    value = value.replace("(", "").replace(")", "")
    return tuple(map(int, value.split(",")))


def parse_cmdline_args_to_config(cfg):
    parser = argparse.ArgumentParser(description="StructDiff configuration")

    parser.add_argument("--run_name", type=str, help="Training run / checkpoint version name.")
    parser.add_argument("--image_name", type=str, help="Training image name or explicit image path.")
    parser.add_argument("--task", type=str, choices=["image"], help="StructDiff keeps only the single-image task.")
    parser.add_argument("--diffusion_timesteps", type=int, help="Number of reverse diffusion steps.")
    parser.add_argument("--network_depth", type=int, help="Number of ARF blocks in ARFNet.")
    parser.add_argument("--network_filters", type=int, help="Base feature width used by ARFNet.")
    parser.add_argument("--available_gpus", type=str, help="CUDA_VISIBLE_DEVICES string.")
    parser.add_argument("--initial_lr", type=float, help="Initial learning rate used for training.")
    parser.add_argument(
        "--resize_max_side",
        type=int,
        help="Automatically resize loaded images and masks so their longest side is at most this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--use_positional_encoding",
        dest="use_positional_encoding",
        action="store_true",
        default=None,
        help="Enable the paper's 3D positional encoding + Fourier features.",
    )
    parser.add_argument("--output_dir", type=str, help="Root directory for saved samples.")
    parser.add_argument(
        "--sample_mode",
        type=str,
        choices=["diverse", "control", "text", "reference", "outpaint"],
        help="Sampling mode: diverse generation, PE-guided control, text guidance, reference guidance, or outpainting.",
    )
    parser.add_argument("--sample_count", type=int, help="Number of samples to generate.")
    parser.add_argument("--batch_size", type=int, help="Batch size used during sampling.")
    parser.add_argument("--sample_size", type=_tuple_of_ints, help="Output image size formatted as H,W.")
    parser.add_argument("--mask_variant", type=str, help="Optional variant mask name or explicit mask path.")

    parser.add_argument(
        "--control_action",
        type=str,
        choices=["none", "shift", "copy_shift", "scale"],
        help="Positional-control transform to apply in control mode. Use 'none' for the unmodified PE.",
    )
    parser.add_argument("--shift_x", type=int, help="Horizontal shift in pixels for PE control.")
    parser.add_argument("--shift_y", type=int, help="Vertical shift in pixels for PE control.")
    parser.add_argument("--scale_factor", type=float, help="Foreground area scaling factor for PE control.")

    parser.add_argument("--text_input", type=str, help="Text prompt for text-guided sampling.")
    parser.add_argument("--text_guidance_strength", type=float, help="CLIP guidance strength.")
    parser.add_argument("--guidance_sub_iters", type=int, help="CLIP refinement iterations per denoising step.")
    parser.add_argument(
        "--guidance_region",
        type=str,
        choices=["foreground", "background", "full"],
        help="Region affected by text guidance.",
    )
    parser.add_argument("--guidance_start_step", type=int, help="Highest timestep that guidance is active for.")
    parser.add_argument("--guidance_stop_step", type=int, help="Lowest timestep that guidance remains active for.")

    parser.add_argument("--reference_image", type=str, help="Reference image name or explicit image path.")
    parser.add_argument(
        "--reference_downsample_factor",
        type=float,
        help="Low-frequency downsample factor used by reference-guided sampling.",
    )
    parser.add_argument("--outpaint_offset", type=_tuple_of_ints, help="Top-left placement of the reference image as X,Y.")

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg
