import math

import torch
from torch import nn

from models.modules import ARFBlock, SinusoidalPosEmb
from models.positional_encoding import LFF


class ARFNet(nn.Module):
    """Single-scale backbone built from ARF blocks and symmetric skip connections."""

    def __init__(self, in_channels=3, out_channels=3, depth=16, filters_per_layer=64):
        super().__init__()

        if isinstance(filters_per_layer, (list, tuple)):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        time_dim = dims[0]
        fourier_hidden_dim = 512

        self.depth = depth
        self.layers = nn.ModuleList()
        self.layers.append(
            ARFBlock(in_channels, dims[0], emb_dim=time_dim, hidden_dim=fourier_hidden_dim, norm=False)
        )

        midpoint = math.ceil(self.depth / 2)
        for index in range(1, midpoint):
            self.layers.append(
                ARFBlock(dims[index - 1], dims[index], emb_dim=time_dim, hidden_dim=fourier_hidden_dim, norm=True)
            )
        for index in range(midpoint, depth):
            self.layers.append(
                ARFBlock(
                    2 * dims[index - 1],
                    dims[index],
                    emb_dim=time_dim,
                    hidden_dim=fourier_hidden_dim,
                    norm=True,
                )
            )

        self.final_conv = nn.Conv2d(dims[depth - 1], out_channels, 1)
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.fourier_encoder = LFF(hidden_size=fourier_hidden_dim)

    def forward(self, x, grid, t, visualize_weights_and_features=False, sample_t=0, step_counter=0, training=False):
        del training

        time_embedding = self.time_encoder(t)
        image = x[:, :3, :, :]
        fourier_embedding = None if grid is None else self.fourier_encoder(grid)

        residuals = []
        midpoint = math.ceil(self.depth / 2)

        for index in range(midpoint):
            block = self.layers[index]
            if index == 0:
                image = block(
                    image,
                    time_embedding,
                    fourier_embedding,
                    visualize_weights_and_features=visualize_weights_and_features,
                    sample_t=sample_t,
                    step_counter=step_counter,
                    grid=grid,
                )
            else:
                image = block(image, time_embedding, fourier_embedding, grid=grid)
            residuals.append(image)

        for index in range(midpoint, self.depth):
            block = self.layers[index]
            image = torch.cat((image, residuals.pop()), dim=1)
            if index == self.depth - 1:
                image = block(
                    image,
                    time_embedding,
                    fourier_embedding,
                    visualize_weights_and_features=visualize_weights_and_features,
                    sample_t=sample_t,
                    step_counter=step_counter,
                    grid=grid,
                )
            else:
                image = block(image, time_embedding, fourier_embedding, grid=grid)

        return self.final_conv(image)
