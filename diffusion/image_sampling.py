import torch

from common_utils.common import two_tuple
from common_utils.resizer import Resizer


def _repeat_to_batch(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1, -1, -1)
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]

    repeat_count = (batch_size + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat(repeat_count, 1, 1, 1)[:batch_size]


def _expanded_mask(mask, batch_size, image_channels=3):
    if mask.shape[1] == 1 and image_channels != 1:
        mask = mask.expand(-1, image_channels, -1, -1)
    return _repeat_to_batch(mask, batch_size)


def build_region_mask(region, foreground_mask, batch_size, image_channels=3):
    region = region.lower()
    if region == "full":
        mask = torch.ones_like(foreground_mask[:, :1])
    elif region == "foreground":
        mask = foreground_mask[:, :1]
    elif region == "background":
        mask = 1 - foreground_mask[:, :1]
    else:
        raise ValueError(f"Unsupported guidance region: {region}")

    return _expanded_mask(mask, batch_size=batch_size, image_channels=image_channels)


def build_outpainting_mask(reference_image, canvas_size, offset=(0, 0), batch_size=1, device=None):
    canvas_height, canvas_width = two_tuple(canvas_size)
    ref_height, ref_width = reference_image.shape[-2:]
    offset_x, offset_y = offset

    if offset_x < 0 or offset_y < 0:
        raise ValueError("outpainting offset must be non-negative.")
    if offset_y + ref_height > canvas_height or offset_x + ref_width > canvas_width:
        raise ValueError(
            f"Reference image of size {(ref_height, ref_width)} does not fit inside "
            f"canvas {(canvas_height, canvas_width)} at offset {(offset_x, offset_y)}."
        )

    mask = torch.zeros((batch_size, 1, canvas_height, canvas_width), dtype=torch.float32, device=device)
    mask[:, :, offset_y:offset_y + ref_height, offset_x:offset_x + ref_width] = 1.0
    return mask


def _build_reference_resizers(batch_size, image_size, downsample_factor, device):
    image_height, image_width = two_tuple(image_size)
    if downsample_factor <= 1:
        raise ValueError("reference_downsample_factor must be greater than 1.")

    shape = (batch_size, 3, image_height, image_width)
    down_shape = (
        batch_size,
        3,
        max(1, int(round(image_height / downsample_factor))),
        max(1, int(round(image_width / downsample_factor))),
    )
    down = Resizer(shape, 1 / downsample_factor).to(device)
    up = Resizer(down_shape, output_shape=(image_height, image_width)).to(device)
    return down, up


@torch.no_grad()
def sample_with_reference_guidance(model, image_size, batch_size, reference_image, start_step, stop_step, downsample_factor):
    image_size = two_tuple(image_size)
    sample_shape = (batch_size, model.channels, image_size[0], image_size[1])
    reference_image = _repeat_to_batch(reference_image.to(model.device), batch_size)
    down, up = _build_reference_resizers(batch_size, image_size, downsample_factor, model.device)

    img = torch.randn(sample_shape, device=model.device)
    timesteps = model.num_timesteps
    for timestep in reversed(range(0, timesteps)):
        img = model.p_sample(img, timestep)
        if stop_step < timestep <= start_step:
            img = img - up(down(img)) + up(down(model.q_sample(reference_image, timestep)))
    return img


def _apply_clip_guidance(x_recon, clip_model, text_embeddings, region_mask, clip_strength, guidance_sub_iters, previous_x_recon=None):
    preserve_weight = 0.05
    guided = x_recon.detach()
    region_mask = region_mask.to(guided.device)

    if previous_x_recon is not None:
        previous_x_recon = previous_x_recon.to(guided.device)
        guided = guided * (1 - region_mask) + (
            (1 - preserve_weight) * previous_x_recon + preserve_weight * guided
        ) * region_mask

    for _ in range(guidance_sub_iters):
        guided_var = guided.detach().requires_grad_(True)
        clip_input = (guided_var + 1) * 0.5
        score = -clip_model.calculate_clip_loss(clip_input, text_embeddings)
        clip_grad = torch.autograd.grad(score, guided_var, create_graph=False)[0]
        masked_grad = clip_grad * region_mask

        image_norm = torch.linalg.vector_norm((guided_var * region_mask).flatten(1), dim=1, keepdim=True)
        grad_norm = torch.linalg.vector_norm(masked_grad.flatten(1), dim=1, keepdim=True).clamp_min(1e-8)
        step_scale = (image_norm / grad_norm).view(-1, 1, 1, 1)

        guided = (guided_var + clip_strength * step_scale * masked_grad).clamp(-1., 1.).detach()

    return guided


def sample_with_text_guidance(
    model,
    image_size,
    batch_size,
    clip_model,
    text_embeddings,
    region_mask,
    clip_strength,
    guidance_sub_iters,
    start_step,
    stop_step,
):
    image_size = two_tuple(image_size)
    sample_shape = (batch_size, model.channels, image_size[0], image_size[1])
    region_mask = _expanded_mask(region_mask.to(model.device), batch_size, image_channels=model.channels)

    img = torch.randn(sample_shape, device=model.device)
    previous_x_recon = None
    timesteps = model.num_timesteps
    for timestep in reversed(range(0, timesteps)):
        with torch.no_grad():
            x_recon = model.predict_x0(img, timestep)

        if stop_step < timestep <= start_step:
            x_recon = _apply_clip_guidance(
                x_recon,
                clip_model=clip_model,
                text_embeddings=text_embeddings,
                region_mask=region_mask,
                clip_strength=clip_strength,
                guidance_sub_iters=guidance_sub_iters,
                previous_x_recon=previous_x_recon,
            )
            previous_x_recon = x_recon.detach()

        with torch.no_grad():
            img = model.p_sample_from_reconstruction(img, x_recon, timestep)

    return img


@torch.no_grad()
def sample_with_outpainting(model, image_size, batch_size, reference_image, outpainting_mask, start_step, stop_step):
    image_size = two_tuple(image_size)
    sample_shape = (batch_size, model.channels, image_size[0], image_size[1])
    reference_image = _repeat_to_batch(reference_image.to(model.device), batch_size)
    outpainting_mask = _expanded_mask(outpainting_mask.to(model.device), batch_size, image_channels=model.channels)

    if reference_image.shape[-2:] != outpainting_mask.shape[-2:]:
        ref_height, ref_width = reference_image.shape[-2:]
        mask_height, mask_width = outpainting_mask.shape[-2:]
        if ref_height > mask_height or ref_width > mask_width:
            raise ValueError("Reference image is larger than the outpainting canvas.")

    img = torch.randn(sample_shape, device=model.device)
    timesteps = model.num_timesteps
    for timestep in reversed(range(0, timesteps)):
        img = model.p_sample(img, timestep)
        if stop_step < timestep <= start_step:
            noisy_reference = model.q_sample(reference_image, timestep)
            padded_reference = torch.zeros_like(img)
            canvas_mask = outpainting_mask[:, :1]
            for batch_index in range(batch_size):
                coords = (canvas_mask[batch_index, 0] > 0.5).nonzero(as_tuple=False)
                if coords.numel() == 0:
                    continue
                y_min, x_min = coords.min(dim=0).values.tolist()
                y_max, x_max = coords.max(dim=0).values.tolist()
                target_height = y_max - y_min + 1
                target_width = x_max - x_min + 1
                if noisy_reference.shape[-2:] != (target_height, target_width):
                    raise ValueError(
                        "Outpainting mask region must match the reference image size. "
                        f"Expected {(noisy_reference.shape[-2], noisy_reference.shape[-1])}, "
                        f"got {(target_height, target_width)}."
                    )
                padded_reference[batch_index, :, y_min:y_max + 1, x_min:x_max + 1] = noisy_reference[batch_index]
            img = padded_reference * outpainting_mask + img * (1 - outpainting_mask)

    return img
