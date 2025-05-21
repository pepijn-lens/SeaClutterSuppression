import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = (displacement, displacement)

    def forward(self, x):
        return torch.roll(x, shifts=self.displacement, dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def get_relative_distances_2d(h, w):
    coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij'), dim=-1)
    coords_flatten = coords.reshape(-1, 2)
    rel_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]
    return rel_coords  # (hw, hw, 2)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size

        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = (self.window_size[0] // 2, self.window_size[1] // 2)
            self.cyclic_shift = CyclicShift((-displacement[0], -displacement[1]))
            self.cyclic_back_shift = CyclicShift(displacement)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances_2d(*self.window_size)
            self.pos_embedding = nn.Parameter(torch.randn(2 * self.window_size[0] - 1, 2 * self.window_size[1] - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1]))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, return_attention=False):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, h, w, c = x.shape
        window_h, window_w = self.window_size
        num_windows_h = h // window_h
        num_windows_w = w // window_w
        heads = self.heads

        qkv = self.to_qkv(x).view(b, h, w, 3, heads, -1 // heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)
        q, k, v = map(lambda t: rearrange(t, 'b h (nh wh) (nw ww) d -> b h (nh nw) (wh ww) d',
                                          nh=num_windows_h, nw=num_windows_w,
                                          wh=window_h, ww=window_w), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            rel_h = self.relative_indices[:, :, 0] + self.window_size[0] - 1
            rel_w = self.relative_indices[:, :, 1] + self.window_size[1] - 1
            dots += self.pos_embedding[rel_h, rel_w]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (nh nw) (wh ww) d -> b (nh wh) (nw ww) (h d)',
                        nh=num_windows_h, nw=num_windows_w, wh=window_h, ww=window_w, h=heads)

        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        if return_attention:
            return out, attn
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(
            dim=dim,
            heads=heads,
            head_dim=head_dim,
            shifted=shifted,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding
        )))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))

    def forward(self, x, **kwargs):
        out = self.attention_block(x, **kwargs)
        # If attention is returned, out is a tuple (out, attn)
        if isinstance(out, tuple):
            out = out[0]
        out = self.mlp_block(out)
        # If attention is returned, out is a tuple (out, attn)
        if isinstance(out, tuple):
            out = out[0]
        return out


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2.'

        self.patch_partition = PatchMerging(in_channels, hidden_dimension, downscaling_factor)

        self.layers = nn.ModuleList()
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(hidden_dimension, num_heads, head_dim, hidden_dimension * 4, False,
                          window_size, relative_pos_embedding),
                SwinBlock(hidden_dimension, num_heads, head_dim, hidden_dimension * 4, True,
                          window_size, relative_pos_embedding)
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=(7, 7),
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(channels, hidden_dim, layers[0], downscaling_factors[0],
                                  heads[0], head_dim, window_size, relative_pos_embedding)
        self.stage2 = StageModule(hidden_dim, hidden_dim * 2, layers[1], downscaling_factors[1],
                                  heads[1], head_dim, window_size, relative_pos_embedding)
        self.stage3 = StageModule(hidden_dim * 2, hidden_dim * 4, layers[2], downscaling_factors[2],
                                  heads[2], head_dim, window_size, relative_pos_embedding)
        self.stage4 = StageModule(hidden_dim * 4, hidden_dim * 8, layers[3], downscaling_factors[3],
                                  heads[3], head_dim, window_size, relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)

class SwinTwoStageModel(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=96, layers=(2, 2), downscaling_factors=(4, 2),
                 num_heads=(3, 6), head_dim=32, window_size=(2, 8), relative_pos_embedding=True, num_classes=1000):
        super().__init__()

        # Stage 1: e.g., 1 → 96 channels
        self.stage1 = StageModule(
            in_channels=in_channels,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=num_heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding
        )

        # Stage 2: e.g., 96 → 192 channels
        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=num_heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding
        )

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)  # (B, C=96, H/4, W/4)
        x = self.stage2(x)  # (B, C=192, H/8, W/8)
        return self.mlp_head(x)

class SwinStageOneModel(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=96, layers=2, downscaling_factor=4, num_heads=3,
                 head_dim=32, window_size=(2,8), relative_pos_embedding=True, num_classes=1000):
        super().__init__()
        self.stage1 = StageModule(in_channels, hidden_dim, layers, downscaling_factor,
                                  num_heads, head_dim, window_size, relative_pos_embedding)
        self.mlp_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)  # Output shape: (B, C, H/4, W/4)
        return self.mlp_head(x)

# Convenience constructors
def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)
