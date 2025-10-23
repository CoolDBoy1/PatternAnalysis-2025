# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        # This is the main sequential part
        self.seq = nn.Sequential(*layers)
        
        # Residual connection if in_ch != out_ch
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.seq(x)
        residual = self.res_conv(x)
        return out + residual

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, use_bn=use_bn, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pooled = self.pool(x_conv)
        return x_conv, x_pooled
    
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True, dropout=0.0):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # after concatenation, number of channels will be out_ch*2
        self.conv = ConvBlock(in_ch, out_ch, use_bn=use_bn, dropout=dropout)

    def forward(self, x, skip):
        x = self.upconv(x)
        # if shapes mismatch due to padding, center-crop skip to x's size
        if x.shape[-2:] != skip.shape[-2:]:
            # simple resize via interpolation on skip
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class ImprovedUNet(nn.Module):
    """Improved 2D UNet suitable for multi-class segmentation.
    - in_channels: input image channels (1 for OASIS grayscale)
    - out_channels: number of segmentation classes
    - features: list controlling channel sizes in encoder
    - dropout: probability of dropping out nodes
    """
    def __init__(self, in_channels=1, out_channels=4, features=[64, 128, 256, 512], dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(prev_ch, f, use_bn=True, dropout=dropout))
            prev_ch = f

        # bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, use_bn=True, dropout=dropout)

        # decoder: reverse features
        rev = features[::-1]
        in_ch = features[-1]*2
        for f in rev:
            # decoder conv expects concatenated channels => in_ch = prev_upconv_out + skip
            self.decoders.append(DecoderBlock(in_ch, f, use_bn=True, dropout=dropout))
            in_ch = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            out, x = enc(x)
            skips.append(out)
        
        x = self.bottleneck(x)

        skips = skips[::-1]
        for idx, dec in enumerate(self.decoders):
            x = dec(x, skips[idx])

        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    # quick shape check
    model = ImprovedUNet(in_channels=1, out_channels=4)
    x = torch.randn(2,1,256,256)
    y = model(x)
    print('output shape', y.shape)