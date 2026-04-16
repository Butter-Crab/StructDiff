import math

import torch


__all__ = [
    "background_grid_like",
    "foreground_mask_from_grid",
    "copy_shift_grid",
    "shift_grid",
    "scale_grid",
]


def background_grid_like(grid):
    if grid is None or grid.shape[1] != 3:
        raise ValueError("Expected a 3-channel StructDiff positional grid.")

    batch_size, _, height, width = grid.shape
    device = grid.device
    dtype = grid.dtype

    x_channel = torch.linspace(-1, 1, width, device=device, dtype=dtype).view(1, 1, 1, width)
    y_channel = torch.linspace(-1, 1, height, device=device, dtype=dtype).view(1, 1, height, 1)
    x_channel = x_channel.expand(batch_size, -1, height, -1)
    y_channel = y_channel.expand(batch_size, -1, -1, width)
    background_channel = torch.ones((batch_size, 1, height, width), device=device, dtype=dtype)
    return torch.cat((x_channel, y_channel, background_channel), dim=1)


def foreground_mask_from_grid(grid):
    if grid is None or grid.shape[1] != 3:
        raise ValueError("Expected a 3-channel StructDiff positional grid.")
    return (grid[:, 2:3] < 0.5).float()


def _apply_destination_copy(source_grid, destination_grid, destination_x, destination_y, source_x, source_y):
    height = destination_grid.shape[-2]
    width = destination_grid.shape[-1]
    valid = (
        (destination_x >= 0)
        & (destination_x < width)
        & (destination_y >= 0)
        & (destination_y < height)
    )
    if not torch.any(valid):
        return destination_grid

    source_x = source_x[valid]
    source_y = source_y[valid]
    destination_x = destination_x[valid]
    destination_y = destination_y[valid]
    destination_grid[:, destination_y, destination_x] = source_grid[:, source_y, source_x]
    return destination_grid


def copy_shift_grid(grid, x_shift=0, y_shift=0):
    mask = foreground_mask_from_grid(grid)
    shifted_grid = grid.clone()

    for batch_index in range(grid.shape[0]):
        source_y, source_x = mask[batch_index, 0].nonzero(as_tuple=True)
        if source_x.numel() == 0:
            continue

        destination_x = source_x + int(x_shift)
        destination_y = source_y + int(y_shift)
        shifted_grid[batch_index] = _apply_destination_copy(
            grid[batch_index],
            shifted_grid[batch_index],
            destination_x,
            destination_y,
            source_x,
            source_y,
        )

    return shifted_grid


def shift_grid(grid, x_shift=0, y_shift=0):
    mask = foreground_mask_from_grid(grid)
    shifted_grid = background_grid_like(grid)

    for batch_index in range(grid.shape[0]):
        source_y, source_x = mask[batch_index, 0].nonzero(as_tuple=True)
        if source_x.numel() == 0:
            continue

        destination_x = source_x + int(x_shift)
        destination_y = source_y + int(y_shift)
        shifted_grid[batch_index] = _apply_destination_copy(
            grid[batch_index],
            shifted_grid[batch_index],
            destination_x,
            destination_y,
            source_x,
            source_y,
        )

    return shifted_grid


def scale_grid(grid, scale_factor=1.0):
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive.")

    mask = foreground_mask_from_grid(grid)
    scaled_grid = background_grid_like(grid)
    linear_scale = math.sqrt(scale_factor)

    for batch_index in range(grid.shape[0]):
        source_y, source_x = mask[batch_index, 0].nonzero(as_tuple=True)
        if source_x.numel() == 0:
            continue

        center_x = (source_x.min().float() + source_x.max().float()) / 2.0
        center_y = (source_y.min().float() + source_y.max().float()) / 2.0
        destination_x = ((source_x.float() - center_x) * linear_scale + center_x).round().long()
        destination_y = ((source_y.float() - center_y) * linear_scale + center_y).round().long()

        scaled_grid[batch_index] = _apply_destination_copy(
            grid[batch_index],
            scaled_grid[batch_index],
            destination_x,
            destination_y,
            source_x,
            source_y,
        )

    return scaled_grid
