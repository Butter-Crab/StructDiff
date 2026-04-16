# Code from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from inspect import isfunction

import torch
from einops import rearrange
from torch import nn, einsum


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class ARFConv(nn.Module):
    def __init__(self, features, branches=4, groups=None, reduction=2, stride=1, min_dim=32):
        """Adaptive receptive field convolution with multi-branch kernel selection."""
        super().__init__()
        groups = features if groups is None else groups
        hidden_dim = max(int(features / reduction), min_dim)
        self.M = branches
        self.features = features
        self.convs = nn.ModuleList()
        for branch_index in range(1, branches + 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3 + branch_index * 2,
                        stride=stride,
                        padding=1 + branch_index,
                        groups=groups,
                    ),
                    nn.BatchNorm2d(features),
                    nn.GELU(),
                )
            )
        self.fc = nn.Linear(features, hidden_dim)
        self.fcs = nn.ModuleList(nn.Linear(hidden_dim, features) for _ in range(branches))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, visualize_weights_and_features=False, sample_t=0, step_counter=0):
        del visualize_weights_and_features, sample_t, step_counter
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        self.attention_weights = attention_vectors
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        self.selected_features = feas * attention_vectors
        fea_v = (feas * attention_vectors).sum(dim=1)

        return fea_v

class ARFBlock(nn.Module):
    """StructDiff block with ARFConv, timestep injection, and optional Fourier features."""

    def __init__(self, dim, dim_out, *, emb_dim=None, hidden_dim=None, mult=3, norm=True, kernel_sizes=(3, 5, 7, 9)):
        del kernel_sizes
        super().__init__()

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, dim)
        ) if exists(emb_dim) else None

        
        self.mlp_ = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )if exists(hidden_dim) else None
        

        self.ds_conv = ARFConv(dim, branches=4, groups=dim, reduction=2, stride=1)
        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None, Fourier=None, visualize_weights_and_features=False, sample_t=0, step_counter=0, grid=None):
        h = self.ds_conv(x,visualize_weights_and_features=visualize_weights_and_features, sample_t=sample_t, step_counter=step_counter)
        
        if exists(self.mlp):
            assert exists(emb), 'time (and possibly frame) emb must be passed in'
            condition = self.mlp(emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        if Fourier is not None:
            b, c, h_, w = Fourier.shape
            Fourier = Fourier.permute(0, 2, 3, 1).reshape(-1, c)
            condition_F = self.mlp_(Fourier)
            condition_F = condition_F.view(b, h_, w, -1).permute(0, 3, 1, 2)
            h = h + condition_F

        if Fourier is None and grid is not None:
            b, c, h_, w = grid.shape
            grid = grid.permute(0, 2, 3, 1).reshape(-1, c)
            condition_F = self.mlp_(grid)
            condition_F = condition_F.view(b, h_, w, -1).permute(0, 3, 1, 2)
            h = h + condition_F


        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)
