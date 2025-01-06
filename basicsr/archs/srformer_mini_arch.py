import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import torch.nn.functional as F

# LightweightPSA ve ConvFFN sınıfları aynı kalıyor...

class LightweightPSA(nn.Module):
    """Lightweight Permuted Self-Attention with Window Attention"""
    def __init__(self, dim, num_heads=4, window_size=8, qkv_bias=True, reduction_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.reduced_dim = max(dim // reduction_ratio, num_heads * 8)
        self.reduced_dim = self.reduced_dim - (self.reduced_dim % num_heads)
        
        self.q = nn.Linear(dim, self.reduced_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.reduced_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        """
        x: (B, N, C), where N = H * W
        H, W: Spatial dimensions of the feature map
        """
        B, N, C = x.shape
        assert H * W == N, "Input tensor dimensions must match H * W"

        # Reshape to (B, H, W, C) for window partitioning
        x = x.view(B, H, W, C)

        # Partition into windows
        x = self.window_partition(x)  # (num_windows*B, window_size*window_size, C)

        # Query, Key, Value computation
        q = self.q(x).reshape(-1, self.window_size**2, self.num_heads, self.reduced_dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(-1, self.window_size**2, self.num_heads, self.reduced_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(-1, self.window_size**2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Attention calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to value
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size**2, C)

        # Merge windows
        x = self.window_reverse(x, H, W)  # (B, H, W, C)

        # Project output
        x = x.view(B, N, C)
        x = self.proj(x)

        return x

    def window_partition(self, x):
        """
        Partition feature map into non-overlapping windows.
        x: (B, H, W, C)
        Returns: (num_windows*B, window_size*window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(
            B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size**2, C)
        return x

    def window_reverse(self, x, H, W):
        """
        Reverse the window partition process to reconstruct feature map.
        x: (num_windows*B, window_size*window_size, C)
        Returns: (B, H, W, C)
        """
        B = x.shape[0] // (H // self.window_size * W // self.window_size)
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
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
class SRFormer_mini(nn.Module):
    def __init__(self, 
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=48,
                 depths=(6, 6),
                 num_heads=(6, 6),
                 window_size=16,
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
                        window_size=window_size,  # Pencere boyutunu aktar
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
        H, W = x.shape[2:]  # Giriş görüntüsünün yüksekliği ve genişliği
        x = self.check_image_size(x)  # Gerekirse padding ekle

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)  # (B, N, C), N = H * W
        
        # Transformer işlemleri
        for layer in self.layers:
            for block in layer:
                for module in block:
                    if isinstance(module, LightweightPSA):
                        x = x + module(x, H, W)  # LightweightPSA'ya H ve W aktarılıyor
                    else:
                        x = x + module(x)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.upsample(x)  # Upsampling uygulanır
        
        x = x / self.img_range + self.mean
        
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

