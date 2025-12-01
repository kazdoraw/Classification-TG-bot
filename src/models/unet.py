"""
U-Net Architecture for Image Segmentation.

Based on the original U-Net paper:
"U-Net: Convolutional Networks for Biomedical Image Segmentation"
Ronneberger et al., 2015

Architecture:
- Encoder (contracting path): 4 downsampling blocks
- Bottleneck: deepest conv block
- Decoder (expansive path): 4 upsampling blocks with skip connections
- Output: segmentation mask with same size as input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU.
    
    Standard building block in U-Net architecture.
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            mid_channels (int, optional): Number of intermediate channels.
                                         If None, uses out_channels
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: MaxPool -> DoubleConv."""
    
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: ConvTranspose -> Concatenate skip connection -> DoubleConv.
    """
    
    def __init__(self, in_channels, out_channels, bilinear=False):
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bilinear (bool): If True, use bilinear upsampling instead of transposed conv
        """
        super().__init__()
        
        # Use bilinear upsampling or transposed convolutions
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Input from previous layer (lower resolution)
            x2: Skip connection from encoder (higher resolution)
            
        Returns:
            Upsampled and concatenated features
        """
        x1 = self.up(x1)
        
        # Handle size mismatch due to pooling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to produce output channels."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Semantic Segmentation.
    
    Features:
    - 4-level encoder-decoder with skip connections
    - Suitable for biomedical image segmentation
    - Maintains spatial information through skip connections
    
    Args:
        n_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        n_classes (int): Number of output classes
        bilinear (bool): Use bilinear upsampling instead of transposed conv
    
    Example:
        >>> model = UNet(n_channels=3, n_classes=1)
        >>> x = torch.randn(1, 3, 512, 512)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 1, 512, 512])
    """
    
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Output segmentation mask of shape (batch, n_classes, height, width)
        """
        # Encoder
        x1 = self.inc(x)    # 64 channels
        x2 = self.down1(x1)  # 128 channels
        x3 = self.down2(x2)  # 256 channels
        x4 = self.down3(x3)  # 512 channels
        x5 = self.down4(x4)  # 1024 (or 512 if bilinear) channels
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # 512 channels
        x = self.up2(x, x3)   # 256 channels
        x = self.up3(x, x2)   # 128 channels
        x = self.up4(x, x1)   # 64 channels
        
        # Output
        logits = self.outc(x)  # n_classes channels
        return logits


if __name__ == "__main__":
    # Test the model
    model = UNet(n_channels=3, n_classes=1)
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
