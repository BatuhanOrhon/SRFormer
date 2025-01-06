import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import torch.nn.functional as F

# LightweightPSA ve ConvFFN sınıfları aynı kalıyor...

class LightweightPSA(nn.Module):
    """Even more lightweight Permuted Self-Attention module"""
    def __init__(self, dim, num_heads=4, qkv_bias=True, reduction_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5
        
        self.reduced_dim = max(dim // reduction_ratio, num_heads * 8)
        self.reduced_dim = self.reduced_dim - (self.reduced_dim % num_heads)
        
        self.q = nn.Linear(dim, self.reduced_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.reduced_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        head_dim = self.reduced_dim // self.num_heads
        
        q = self.q(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class ConvFFN(nn.Module):
    """Further simplified ConvFFN"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        reduced_hidden = max(hidden_features // 4, in_features)
        
        self.fc1 = nn.Linear(in_features, reduced_hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(reduced_hidden, out_features)

    def forward(self, x, x_size=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

@ARCH_REGISTRY.register()
class SRFormer_global_mini(nn.Module):
    def __init__(self, 
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=48,
                 depths=(4, 4),
                 num_heads=(4, 4),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 upscale=4,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv'):
        super().__init__()
        
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        
        self.img_range = img_range
        self.in_chans = in_chans
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
            
        self.upscale = upscale
        self.window_size = window_size

        # Transformer stages
        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        for i_layer in range(len(depths)):
            layer = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    LightweightPSA(
                        dim=embed_dim,
                        num_heads=num_heads[i_layer],
                        qkv_bias=qkv_bias
                    ),
                    nn.LayerNorm(embed_dim),
                    ConvFFN(
                        in_features=embed_dim,
                        hidden_features=int(embed_dim * mlp_ratio)
                    )
                ) for _ in range(depths[i_layer])
            ])
            self.layers.append(layer)

        # Simplified upsampling
        if upsampler == 'pixelshuffle':
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, self.in_chans * (self.upscale ** 2), 3, 1, 1),
                nn.PixelShuffle(upscale)
            )
        elif upsampler == 'pixelshuffledirect':
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, (upscale ** 2) * in_chans, 3, 1, 1),
                nn.PixelShuffle(upscale)
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        H, W = x.shape[2:]
        
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        B, C, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)
        
        # Process through transformer stages
        for layer in self.layers:
            for block in layer:
                x = x + block(x)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.upsample(x)  # This now outputs correct number of channels
        
        x = x / self.img_range + self.mean
        return x[:, :, :H*self.upscale, :W*self.upscale]

