# modules.py
import torch
import torch.nn as nn
from timm.layers import DropPath

# Permute for LayerNorm: moves channel dim to last for nn.LayerNorm
class PermuteForLN(nn.Module):
    def forward(self, x):
        # [B, C, D, H, W] -> [B, D, H, W, C]
        return x.permute(0, 2, 3, 4, 1)

# Permute back to original 3D CNN format: [B, D, H, W, C] -> [B, C, D, H, W]
class PermuteBack(nn.Module):
    def forward(self, x):
        # [B, D, H, W, C] -> [B, C, D, H, W]
        return x.permute(0, 4, 1, 2, 3)
    
class ConvNeXtBlock3D(nn.Module):
    """A single ConvNeXt block for 3D volumes."""
    def __init__(self, dim, drop_rate=0, layer_scale_init_value=1e-6):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim),  # depthwise
            PermuteForLN(),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),   
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, 1, 1, dim)), requires_grad=True)
        self.back = nn.Sequential(
            PermuteBack(),
            DropPath(drop_rate),
        )

    def forward(self, x):
        return x + self.back(self.gamma * self.block(x))
    
class ConvNeXt(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, depths=[2, 2, 4, 2], dims=[16, 32, 64, 128], drop_rate=0):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        # stem: initial downsampling layer
        # Kernel size (1,4,4) reduces spatial dimensions, keeps depth intact
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(1,4,4), stride=(1,4,4)),  # downsample
            PermuteForLN(),
            nn.LayerNorm(dims[0], eps=1e-6),
            PermuteBack(),
        )
        self.downsample_layers.append(stem)
        
        # dp_rates: drop path rate for each block, linearly increasing from 0 to drop_rate
        dp_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(depths))] 
        
        cur = 0
        for i in range(len(depths)):
            # Downsampling between stages: halves spatial resolution, increases channels
            stage = nn.Sequential(
                *[ConvNeXtBlock3D(dims[i], drop_rate=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            
            if i < len(depths) - 1:
                # remaining downsampling layers
                downsample_layer = nn.Sequential(
                    PermuteForLN(),
                    nn.LayerNorm(dims[i], eps=1e-6),
                    PermuteBack(),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,2,2), stride=(1,2,2)),  # downsample
                )
                self.downsample_layers.append(downsample_layer)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):   
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = ConvNeXt()
    x = torch.randn(2, 1, 64, 224, 224)  # batch of 2, 1 channel, 64 slices, 224x224
    y = model(x)
    print(y.shape)