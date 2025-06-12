import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Handle relative imports
try:
    from .swin_transformer import SwinBlock, _to_2tuple
except ImportError:
    # If running as standalone script
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from swin_transformer import SwinBlock, _to_2tuple

class PatchEmbedding(nn.Module):
    """Image to Patch Embedding for U-Net"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # B embed_dim Wh Ww
        x = x.flatten(2).transpose(1, 2)  # B Wh*Ww embed_dim
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMergingUNet(nn.Module):
    """Patch merging layer for U-Net downsampling"""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpanding(nn.Module):
    """Patch expanding layer for upsampling in decoder"""
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x = self.norm(x)

        return x

class FinalPatchExpand(nn.Module):
    """Final patch expanding layer"""
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x = self.norm(x)

        return x

class SwinEncoderLayer(nn.Module):
    """Swin Transformer Encoder Layer with downsampling"""
    def __init__(self, input_resolution, dim, depth, num_heads, window_size=7, 
                 shift_size=0, mlp_ratio=4., drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Create list of Swin blocks based on depth
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinBlock(
                    dim=dim,
                    heads=num_heads,
                    head_dim=dim // num_heads,
                    mlp_dim=int(dim * mlp_ratio),
                    shifted=(i % 2 == 1),  # Alternate between regular and shifted windows
                    window_size=window_size,
                    relative_pos_embedding=True
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """
        Input: x with shape (B, H, W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, "input channel dim does not match layer dim"

        # reshape to (B, H, W, C) format for Swin blocks
        x = x.view(B, H, W, C)
        
        # apply Swin transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # reshape back to (B, L, C)
        x = x.view(B, H * W, C)

        # downsample
        if self.downsample is not None:
            x = self.downsample(x)

        return x

class SwinDecoderLayer(nn.Module):
    """Swin Transformer Decoder Layer with upsampling"""
    def __init__(self, input_resolution, dim, depth, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # upsample layer
        if upsample is not None:
            if hasattr(upsample, '__call__') and not isinstance(upsample, type):
                # It's an already instantiated object
                self.upsample = upsample
            else:
                # It's a class, instantiate it
                self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

        # Create list of Swin blocks based on depth
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinBlock(
                    dim=dim,
                    heads=num_heads,
                    head_dim=dim // num_heads,
                    mlp_dim=int(dim * mlp_ratio),
                    shifted=(i % 2 == 1),  # Alternate between regular and shifted windows
                    window_size=window_size,
                    relative_pos_embedding=True
                )
            )

    def forward(self, x, skip=None):
        """
        Input: x with shape (B, L, C)
        """
        # upsample first
        if self.upsample is not None:
            x = self.upsample(x)

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # concatenate with skip connection if provided
        if skip is not None:
            # Ensure skip connection has the same spatial dimensions
            skip_B, skip_L, skip_C = skip.shape
            if skip_L != L:
                # If dimensions don't match, we need to handle this
                skip = skip.view(skip_B, int(skip_L**0.5), int(skip_L**0.5), skip_C)
                skip = F.interpolate(skip.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False)
                skip = skip.permute(0, 2, 3, 1).view(skip_B, H*W, skip_C)
            
            x = torch.cat([x, skip], dim=-1)
            # Project concatenated features back to expected dimension
            x = nn.Linear(C + skip_C, C, device=x.device)(x)

        # reshape to (B, H, W, C) format for Swin blocks
        x = x.view(B, H, W, C)
        
        # apply Swin transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # reshape back to (B, L, C)
        x = x.view(B, H * W, C)

        return x

class SwinUNet(nn.Module):
    """Swin Transformer U-Net for segmentation"""
    def __init__(self, img_size=128, patch_size=4, in_chans=3, num_classes=1,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, patches_resolution[0] * patches_resolution[1], embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_resolution = (patches_resolution[0] // (2 ** i_layer),
                              patches_resolution[1] // (2 ** i_layer))
            layer = SwinEncoderLayer(
                input_resolution=layer_resolution,
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMergingUNet if (i_layer < self.num_layers - 1) else None
            )
            self.encoder_layers.append(layer)

        # build decoder layers
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            layer_resolution = (patches_resolution[0] // (2 ** i_layer),
                              patches_resolution[1] // (2 ** i_layer))
            
            # Calculate current layer dimension (output dimension)
            current_dim = int(embed_dim * 2 ** i_layer)
            
            if i_layer == self.num_layers - 1:
                # First decoder layer (no skip connection, no upsampling)
                # Input and output have the same dimension
                layer = SwinDecoderLayer(
                    input_resolution=layer_resolution,
                    dim=current_dim,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    upsample=None
                )
            else:
                # Decoder layers with skip connections and upsampling
                # Input dimension is from previous layer (higher dimension)
                input_dim = int(embed_dim * 2 ** (i_layer + 1))
                
                # Create custom upsampling layer that goes from input_dim to current_dim
                class CustomPatchExpanding(nn.Module):
                    def __init__(self, input_resolution, output_resolution, input_dim, output_dim, norm_layer=nn.LayerNorm):
                        super().__init__()
                        self.input_resolution = input_resolution
                        self.output_resolution = output_resolution
                        self.input_dim = input_dim
                        self.output_dim = output_dim
                        self.expand = nn.Linear(input_dim, input_dim * 2, bias=False)
                        self.norm = norm_layer(output_dim)

                    def forward(self, x):
                        from einops import rearrange
                        H, W = self.input_resolution
                        x = self.expand(x)  # input_dim -> input_dim * 2
                        B, L, C = x.shape
                        assert L == H * W, f"input feature has wrong size: got L={L}, expected H*W={H*W}"

                        x = x.view(B, H, W, C)
                        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
                        x = x.view(B, -1, C//4)
                        x = self.norm(x)
                        return x
                
                # Input resolution is from previous layer (smaller)
                input_resolution = (patches_resolution[0] // (2 ** (i_layer + 1)),
                                  patches_resolution[1] // (2 ** (i_layer + 1)))
                
                upsample_layer = CustomPatchExpanding(
                    input_resolution=input_resolution,
                    output_resolution=layer_resolution,
                    input_dim=input_dim,
                    output_dim=current_dim,
                    norm_layer=norm_layer
                )
                
                layer = SwinDecoderLayer(
                    input_resolution=layer_resolution,
                    dim=current_dim,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    upsample=upsample_layer
                )
            self.decoder_layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)

        # final patch expanding and output
        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand(
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                dim_scale=patch_size,
                dim=embed_dim
            )
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # patch embedding
        x = self.patch_embed(x)  # (B, L, embed_dim) where L = (H/patch_size) * (W/patch_size)
        
        # absolute position embedding
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # encoder
        encoder_outputs = []
        for layer in self.encoder_layers:
            encoder_outputs.append(x)
            x = layer(x)

        # decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                # First decoder layer (deepest)
                x = layer(x)
            else:
                # Use skip connection from encoder
                skip_idx = self.num_layers - 1 - i
                skip = encoder_outputs[skip_idx]
                x = layer(x, skip)

        x = self.norm_up(x)  # B L C

        # final upsample
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, H, W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
            x = torch.sigmoid(x)  # Sigmoid activation for segmentation

        return x

def swin_unet_tiny(pretrained=False, **kwargs):
    """Swin-UNet-Tiny model"""
    model = SwinUNet(
        patch_size=4,
        embed_dim=64,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,
        mlp_ratio=4,
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model

def swin_unet_small(pretrained=False, **kwargs):
    """Swin-UNet-Small model"""
    model = SwinUNet(
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        drop_rate=0.0,
        drop_path_rate=0.2,
        **kwargs
    )
    return model

def swin_unet_base(pretrained=False, **kwargs):
    """Swin-UNet-Base model"""
    model = SwinUNet(
        patch_size=4,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=4,
        mlp_ratio=4,
        drop_rate=0.0,
        drop_path_rate=0.2,
        **kwargs
    )
    return model

# Radar-specific Swin U-Net for your sea clutter dataset
def radar_swin_unet(img_size=128, in_chans=1, num_classes=1, **kwargs):
    """Radar-specific Swin U-Net optimized for 128x128 range-doppler maps"""
    model = SwinUNet(
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        **kwargs
    )
    return model

if __name__ == "__main__":
    # Test the model
    model = swin_unet_tiny(img_size=128, in_chans=1, num_classes=1)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Radar Swin U-Net: {trainable_params:,} trainable parameters')
    
    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    output = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {output.shape}')
