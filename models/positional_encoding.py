import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding3D(nn.Module):
    """
    3D positional encoding used in StructDiff.

    Each pixel is encoded by normalized x/y coordinates and a foreground-background indicator.
    """

    def forward(self, x, mask, fg_bg=True):
        b, _, h, w = x.shape
        device = x.device

        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).repeat(b, 1, h, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).repeat(b, 1, 1, w)
        out = torch.cat((x_channel, y_channel), dim=1).to(x)

        if not fg_bg:
            return out

        if mask is None:
            raise ValueError("A foreground mask is required for 3D positional encoding.")

        mask = mask[:, 0:1, :, :].to(device)
        mask = (mask > 0).float()
        fg_bg_channel = 1 - mask
        return torch.cat((out, fg_bg_channel), dim=1).to(x)


class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        return self.conv(x)


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class LFF(nn.Module):
    """
    Learnable Fourier features used to embed the 3D positional encoding.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.ffm = ConLinear(3, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        return self.activation(self.ffm(x))
