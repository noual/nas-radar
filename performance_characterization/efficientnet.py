"""
EfficientNet-based Segmentation Network.

This module implements an EfficientNet-style encoder with a UNet-like decoder
for image segmentation tasks. The architecture uses Mobile Inverted Bottleneck
Convolutions (MBConv) with Squeeze-and-Excitation (SE) blocks.

Adapted for single-channel radar image segmentation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Adaptively recalibrates channel-wise feature responses by explicitly 
    modeling interdependencies between channels.
    """
    
    def __init__(self, in_channels: int, squeeze_ratio: int = 4):
        super().__init__()
        squeezed_channels = max(1, in_channels // squeeze_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeezed_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeezed_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block.
    
    This is the core building block of EfficientNet, featuring:
    - Expansion with 1x1 conv
    - Depthwise separable convolution
    - Squeeze-and-Excitation
    - Projection with 1x1 conv
    - Skip connection (when stride=1 and in_channels==out_channels)
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for depthwise conv (3 or 5).
        stride: Stride for depthwise conv.
        expand_ratio: Expansion ratio for the block.
        se_ratio: Squeeze-Excitation ratio (0 to disable SE).
        drop_connect_rate: Drop connect rate for stochastic depth.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        
        # Expansion phase (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride=stride, padding=padding, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            squeeze_channels = max(1, int(in_channels * se_ratio))
            layers.append(SqueezeExcitation(expanded_channels, int(expanded_channels / squeeze_channels)))
        
        # Projection phase (1x1 conv)
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth (drop connect) during training."""
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep_prob = 1 - self.drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob + torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_mask = random_tensor.floor()
        return x / keep_prob * binary_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = self._drop_connect(out) + x
        return out


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-style encoder for feature extraction.
    
    Produces multi-scale features for use in segmentation decoder.
    
    Args:
        in_channels: Number of input channels.
        width_mult: Width multiplier for channel scaling.
        depth_mult: Depth multiplier for block count scaling.
        drop_connect_rate: Base drop connect rate.
    """
    
    # Block configurations: (expand_ratio, channels, num_blocks, stride, kernel_size)
    # Based on EfficientNet-B0 configuration, adapted for segmentation
    DEFAULT_BLOCKS = [
        (1, 16, 1, 1, 3),   # Stage 1: No downsampling for higher resolution
        (6, 24, 2, 2, 3),   # Stage 2: Downsample
        (6, 40, 2, 2, 5),   # Stage 3: Downsample
        (6, 80, 3, 2, 3),   # Stage 4: Downsample
        (6, 112, 3, 1, 5),  # Stage 5: No downsampling
        (6, 192, 4, 2, 5),  # Stage 6: Downsample
        (6, 320, 1, 1, 3),  # Stage 7: No downsampling
    ]
    
    def __init__(
        self,
        in_channels: int = 1,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        drop_connect_rate: float = 0.2
    ):
        super().__init__()
        
        def scale_channels(channels: int) -> int:
            return int(math.ceil(channels * width_mult / 8) * 8)
        
        def scale_depth(num_blocks: int) -> int:
            return int(math.ceil(num_blocks * depth_mult))
        
        # Stem
        stem_channels = scale_channels(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        self.stage_channels = [stem_channels]  # Track output channels for decoder
        
        prev_channels = stem_channels
        total_blocks = sum(scale_depth(b[2]) for b in self.DEFAULT_BLOCKS)
        block_idx = 0
        
        for expand_ratio, channels, num_blocks, stride, kernel_size in self.DEFAULT_BLOCKS:
            out_channels = scale_channels(channels)
            num_blocks = scale_depth(num_blocks)
            
            stage_blocks = []
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                stage_blocks.append(MBConvBlock(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    expand_ratio=expand_ratio,
                    drop_connect_rate=drop_rate
                ))
                prev_channels = out_channels
                block_idx += 1
            
            self.stages.append(nn.Sequential(*stage_blocks))
            self.stage_channels.append(out_channels)
        
        self.out_channels = prev_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features.
        
        Returns features at different scales for skip connections.
        """
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features


class DecoderBlock(nn.Module):
    """
    Decoder block for upsampling and feature fusion.
    
    Args:
        in_channels: Number of input channels (from previous decoder stage).
        skip_channels: Number of channels from skip connection.
        out_channels: Number of output channels.
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Fusion convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        
        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class EfficientNetSegmentation(nn.Module):
    """
    EfficientNet-based Segmentation Network.
    
    Uses EfficientNet encoder with a UNet-style decoder for pixel-wise
    segmentation. Designed for single-channel radar image segmentation.
    
    Args:
        in_channels: Number of input channels (1 for grayscale radar).
        out_channels: Number of output channels (1 for binary segmentation).
        encoder_variant: EfficientNet variant ('b0', 'b1', 'b2', etc.).
        decoder_channels: List of channel counts for decoder stages.
        drop_connect_rate: Drop connect rate for encoder.
    
    Example:
        >>> model = EfficientNetSegmentation(in_channels=1, out_channels=1)
        >>> x = torch.randn(1, 1, 256, 256)
        >>> output = model(x)  # Shape: (1, 1, 256, 256)
    """
    
    # Width and depth multipliers for EfficientNet variants
    VARIANT_PARAMS = {
        'b0': (1.0, 1.0),
        'b1': (1.0, 1.1),
        'b2': (1.1, 1.2),
        'b3': (1.2, 1.4),
        'b4': (1.4, 1.8),
        'b5': (1.6, 2.2),
        'b6': (1.8, 2.6),
        'b7': (2.0, 3.1),
    }
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_variant: str = 'b0',
        decoder_channels: List[int] = None,
        drop_connect_rate: float = 0.2
    ):
        super().__init__()
        
        # Get variant parameters
        width_mult, depth_mult = self.VARIANT_PARAMS.get(encoder_variant, (1.0, 1.0))
        
        # Build encoder
        self.encoder = EfficientNetEncoder(
            in_channels=in_channels,
            width_mult=width_mult,
            depth_mult=depth_mult,
            drop_connect_rate=drop_connect_rate
        )
        
        # Get encoder output channels at each stage
        # stage_channels = [stem, stage1, stage2, stage3, stage4, stage5, stage6, stage7]
        # Indices:          [0,    1,      2,      3,      4,      5,      6,      7]
        encoder_channels = self.encoder.stage_channels
        
        # Default decoder channels (progressively reducing)
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]
        
        # Select which encoder stages to use for skip connections
        # We want to match spatial resolutions - use stages that downsample (stride=2)
        # Stages with stride=2: stage2 (idx 2), stage3 (idx 3), stage4 (idx 4), stage6 (idx 6)
        # The encoder output is from stage7 (idx 7), which has same resolution as stage6
        # So we go: stage7 -> upsample -> cat(stage4) -> upsample -> cat(stage3) -> upsample -> cat(stage2) -> upsample -> cat(stem)
        
        # Skip connection channels in order of use (from deep to shallow)
        # First decoder: no skip (just upsample encoder output)
        # Then: stage5+6 combined (same resolution), stage4, stage3, stage2, stem
        self.skip_indices = [6, 4, 3, 2, 0]  # Encoder feature indices for skip connections
        
        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        
        # First decoder block: takes encoder output, no skip connection
        prev_channels = encoder_channels[-1]  # Last encoder output (stage 7)
        
        for i, dec_channels in enumerate(decoder_channels):
            # Get skip connection channels
            if i < len(self.skip_indices):
                skip_ch = encoder_channels[self.skip_indices[i]]
            else:
                skip_ch = 0
            
            self.decoder_blocks.append(DecoderBlock(prev_channels, skip_ch, dec_channels))
            prev_channels = dec_channels
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], out_channels, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Segmentation logits of shape (B, out_channels, H, W).
        """
        input_size = x.shape[2:]
        
        # Encoder: get multi-scale features
        # features = [stem_out, stage1_out, stage2_out, ..., stage7_out]
        features = self.encoder(x)
        
        # Start with the deepest encoder output
        x = features[-1]
        
        # Get skip features in order of use (deep to shallow)
        # self.skip_indices = [6, 4, 3, 2, 0] -> features at these indices
        skip_features = [features[idx] for idx in self.skip_indices]
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_features[i] if i < len(skip_features) else None
            x = decoder_block(x, skip)
        
        # Ensure output matches input size
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return self.seg_head(x)
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience functions for creating different variants
def efficientnet_b0_seg(in_channels: int = 1, out_channels: int = 1) -> EfficientNetSegmentation:
    """Create EfficientNet-B0 based segmentation model."""
    return EfficientNetSegmentation(in_channels, out_channels, encoder_variant='b0')


def efficientnet_b1_seg(in_channels: int = 1, out_channels: int = 1) -> EfficientNetSegmentation:
    """Create EfficientNet-B1 based segmentation model."""
    return EfficientNetSegmentation(in_channels, out_channels, encoder_variant='b1')


def efficientnet_b2_seg(in_channels: int = 1, out_channels: int = 1) -> EfficientNetSegmentation:
    """Create EfficientNet-B2 based segmentation model."""
    return EfficientNetSegmentation(in_channels, out_channels, encoder_variant='b2')


def efficientnet_b3_seg(in_channels: int = 1, out_channels: int = 1) -> EfficientNetSegmentation:
    """Create EfficientNet-B3 based segmentation model."""
    return EfficientNetSegmentation(in_channels, out_channels, encoder_variant='b3')


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = EfficientNetSegmentation(
        in_channels=1,
        out_channels=1,
        encoder_variant='b0'
    ).to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Benchmark
    import time
    model.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    n_runs = 100
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"Average inference time: {elapsed / n_runs * 1000:.2f} ms")
