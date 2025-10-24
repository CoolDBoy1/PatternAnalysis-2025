# modules.py
import torch
import torch.nn as nn

class PermuteForLN(nn.Module):
    def forward(self, x):
        # [B, C, D, H, W] -> [B, D, H, W, C]
        return x.permute(0, 2, 3, 4, 1)

class PermuteBack(nn.Module):
    def forward(self, x):
        # [B, D, H, W, C] -> [B, C, D, H, W]
        return x.permute(0, 4, 1, 2, 3)
    
class ConvNeXtBlock3D(nn.Module):
    """A single ConvNeXt block for 3D volumes."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim),  # depthwise
            PermuteForLN(),
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            PermuteBack()
        )

    def forward(self, x):
        return x + self.block(x)
    
class ConvNeXt(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, depths=[2,2,6,2], dims=[32,64,128,256]):
        super().__init__()
        self.stem = nn.Conv3d(in_chans, dims[0], kernel_size=(1,4,4), stride=(1,4,4))  # downsample
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvNeXtBlock3D(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            if i < len(depths) - 1:
                self.stages.append(nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,2,2), stride=(1,2,2)))  # downsample

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = ConvNeXt()
    x = torch.randn(2, 1, 64, 224, 224)  # batch of 2, 1 channel, 64 slices, 224x224
    y = model(x)
    print(y.shape)