import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _to_2tuple(x):
    """Ensure *x* is a pair (h, w)."""
    return (x, x) if isinstance(x, int) else tuple(x)


# -----------------------------------------------------------------------------
# Core building blocks
# -----------------------------------------------------------------------------

class CyclicShift(nn.Module):
    """Cyclically shift a feature map in height (dim=1) and width (dim=2)."""

    def __init__(self, displacement):
        super().__init__()
        # Accept either an int or tuple[int, int]
        disp = displacement if isinstance(displacement, (tuple, list)) else (displacement, displacement)
        self.register_buffer("displacement", torch.tensor(disp, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C)
        dh, dw = map(int, self.displacement.tolist())
        return torch.roll(x, shifts=(dh, dw), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # If input is a tuple, operate on the first element, pass through the rest
        if isinstance(x, tuple):
            out = self.fn(x[0], **kwargs)
            if isinstance(out, tuple):
                return (out[0] + x[0], *out[1:])
            return out + x[0]
        else:
            out = self.fn(x, **kwargs)
            if isinstance(out, tuple):
                return (out[0] + x, *out[1:])
            return out + x



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if isinstance(x, tuple):
            normed = self.norm(x[0])
            out = self.fn(normed, **kwargs)
            if isinstance(out, tuple):
                return (out[0], *out[1:])
            return out
        else:
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

# -----------------------------------------------------------------------------
# Window‑level helpers (rectangular‑aware)
# -----------------------------------------------------------------------------

def create_mask(window_size, displacement, upper_lower, left_right):
    """Generate attention mask for shifted windows (rectangular-aware)."""
    h, w = _to_2tuple(window_size)
    dh, dw = _to_2tuple(displacement)

    mask = torch.zeros(h * w, h * w)

    # Top⇄bottom interaction masking
    if upper_lower and dh > 0:
        mask[-dh * w :, : -dh * w] = float("-inf")
        mask[: -dh * w, -dh * w :] = float("-inf")

    # Left⇄right interaction masking
    if left_right and dw > 0:
        mask = rearrange(mask, "(h1 w1) (h2 w2) -> h1 w1 h2 w2", h1=h, w1=w, h2=h, w2=w)
        mask[:, -dw:, :, :-dw] = float("-inf")
        mask[:, :-dw, :, -dw:] = float("-inf")
        mask = rearrange(mask, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")

    return mask


def get_relative_distances(window_size):
    """Compute relative offsets (dy, dx) for every pair of positions in a window."""
    h, w = _to_2tuple(window_size)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack((yy, xx), axis=-1).reshape(-1, 2)  # (N, 2)
    coords = torch.tensor(coords, dtype=torch.long)
    rel = coords[None, :, :] - coords[:, None, :]  # (N, N, 2)
    return rel  # dy, dx

# -----------------------------------------------------------------------------
# Window Attention (rectangular‑aware)
# -----------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.window_size = _to_2tuple(window_size)
        Wh, Ww = self.window_size
        inner_dim = head_dim * heads

        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            disp = (Wh // 2, Ww // 2)
            self.cyclic_shift = CyclicShift(tuple(-d for d in disp))
            self.cyclic_back_shift = CyclicShift(disp)
            self.upper_lower_mask = nn.Parameter(
                create_mask(self.window_size, disp, upper_lower=True, left_right=False), requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(self.window_size, disp, upper_lower=False, left_right=True), requires_grad=False
            )

        # q, k, v projections
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Relative or absolute positional bias
        if self.relative_pos_embedding:
            rel = get_relative_distances(self.window_size)
            rel[:, :, 0] += Wh - 1  # shift to positive
            rel[:, :, 1] += Ww - 1
            self.register_buffer("relative_indices", rel, persistent=False)
            self.pos_embedding = nn.Parameter(torch.randn(2 * Wh - 1, 2 * Ww - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(Wh * Ww, Wh * Ww))

        self.to_out = nn.Linear(inner_dim, dim)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, x, return_attention=False):  # (B, H, W, C)
        if self.shifted:
            x = self.cyclic_shift(x)

        B, H, W, _ = x.shape
        Wh, Ww = self.window_size
        assert H % Wh == 0 and W % Ww == 0, "Input dimensions must be divisible by the window size."

        nh = H // Wh  # number of windows along height
        nw = W // Ww  # number of windows along width

        # Project to q, k, v and partition into windows ----------------------------------
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            rearrange(t, "b (nh h) (nw w) (heads d) -> b heads (nh nw) (h w) d",
                      nh=nh, nw=nw, h=Wh, w=Ww, heads=self.heads)
            for t in qkv
        ]  # each: (B, heads, num_windows, window_area, head_dim)

        # Scaled dot‑product attention ----------------------------------------------------
        dots = einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale  # (B, heads, num_windows, i, j)

        # Positional bias -----------------------------------------------------------------
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        # Shift masks ---------------------------------------------------------------------
        if self.shifted:
            # Last row of windows interacts with first row (upper⇄lower)
            dots[:, :, -nw:] += self.upper_lower_mask
            # Each row: last window interacts with first (left⇄right)
            dots[:, :, nw - 1 :: nw] += self.left_right_mask

        attn = dots.softmax(dim=-1)
        # Aggregate -----------------------------------------------------------------------
        out = einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        out = rearrange(
            out,
            "b heads (nh nw) (h w) d -> b (nh h) (nw w) (heads d)",
            nh=nh,
            nw=nw,
            h=Wh,
            w=Ww,
            heads=self.heads,
        )
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)

        if return_attention:
            return out, attn
        return out

# -----------------------------------------------------------------------------
# Swin block / stages (unchanged except rectangular support plumbing)
# -----------------------------------------------------------------------------

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(
            PreNorm(
                dim,
                WindowAttention(
                    dim=dim,
                    heads=heads,
                    head_dim=head_dim,
                    shifted=shifted,
                    window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding,
                )
            )
        )
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        hidden_dimension,
        layers,
        downscaling_factor,
        num_heads,
        head_dim,
        window_size,
        relative_pos_embedding,
    ):
        super().__init__()
        assert layers % 2 == 0, "Stage layers need to be divisible by 2 for regular and shifted block."

        self.patch_partition = PatchMerging(
            in_channels=in_channels, out_channels=hidden_dimension, downscaling_factor=downscaling_factor
        )
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(
                nn.ModuleList(
                    [
                        SwinBlock(
                            dim=hidden_dimension,
                            heads=num_heads,
                            head_dim=head_dim,
                            mlp_dim=hidden_dimension * 4,
                            shifted=False,
                            window_size=window_size,
                            relative_pos_embedding=relative_pos_embedding,
                        ),
                        SwinBlock(
                            dim=hidden_dimension,
                            heads=num_heads,
                            head_dim=head_dim,
                            mlp_dim=hidden_dimension * 4,
                            shifted=True,
                            window_size=window_size,
                            relative_pos_embedding=relative_pos_embedding,
                        ),
                    ]
                )
            )

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        # (B, H, W, C) -> (B, C, H, W) for next stage
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim,
        layers,
        heads,
        channels=3,
        num_classes=1000,
        head_dim=32,
        window_size=7,
        downscaling_factors=(4, 2, 2, 2),
        relative_pos_embedding=True,
    ):
        super().__init__()

        self.stage1 = StageModule(
            in_channels=channels,
            hidden_dimension=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dimension=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes),
        )

    # ---------------------------------------------------------------------
    def forward(self, img):  # (B, C, H, W)
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])  # global average pool
        return self.mlp_head(x)


# Convenience factory ----------------------------------------------------------

def swin_t(
    *,
    hidden_dim: int = 96,
    layers: tuple = (2, 2, 6, 2),
    heads: tuple = (3, 6, 12, 24),
    window_size=7,
    **kwargs,
):
    """Create Swin-T with optional *window_size* as int or (h, w)."""
    return SwinTransformer(
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        window_size=window_size,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Radar Swin Transformer – single‑stage, local‑only
# -----------------------------------------------------------------------------
# This variant keeps just the first Swin stage (local attention only) and fixes
# the architectural hyper‑parameters requested for radar imagery:
#   • patch size: 8×8  (downscaling factor = 8)
#   • hidden dim: 128
#   • window size: (2, 8)
#   • heads: 4  (head_dim=32 so 32×4 = 128)
#   • depth: 4 Swin blocks  → 2 regular/shifted pairs
# -----------------------------------------------------------------------------

class RadarSwinTransformer(nn.Module):
    def __init__(self,
                 *,
                 in_channels=1,
                 num_classes=10,
                 hidden_dim=128,
                 layers=4,
                 heads=4,
                 head_dim=32,
                 window_size=(2, 8),
                 patch_size=8,
                 relative_pos_embedding=True):
        super().__init__()

        # backbone stage
        self.stage1 = StageModule(
            in_channels=in_channels,
            hidden_dimension=hidden_dim,
            layers=layers,
            downscaling_factor=patch_size,
            num_heads=heads,
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

        # head: mirror Microsoft Swin implementation
        self.norm = nn.LayerNorm(hidden_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.stage1(x)             # -> (B, C=hidden_dim, H', W')

        # reshape to sequence of tokens: (B, C, H'*W') -> (B, H'*W', C)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)

        # apply layer norm on channel dimension for each token
        x = self.norm(x)              # -> (B, L, C)

        # global average pooling over tokens
        x = self.avgpool(x.transpose(1, 2)).squeeze(-1)  # -> (B, C)

        # final classification head
        x = self.head(x)              # -> (B, num_classes)
        return x


# Convenience factory ----------------------------------------------------------

def radar_swin_t(
    in_channels: int = 3,
    num_classes: int = 1000,
    hidden_dim=128,
    layers=4,
    heads=4,
    head_dim=32,
    window_size=(2, 8),
    patch_size=8
) -> nn.Module:
    return RadarSwinTransformer(in_channels=in_channels, num_classes=num_classes, hidden_dim=hidden_dim, layers=layers,
                                heads=heads, head_dim=head_dim, window_size=window_size, patch_size=patch_size)

if __name__ == "__main__":
    model = radar_swin_t(in_channels=1, num_classes=6)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{trainable_params:,} trainable parameters')
