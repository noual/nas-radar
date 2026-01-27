# import math
# import torch
# import time
# import torch.nn as nn
# import torch.nn.functional as F

import math
import torch
import torch.nn as nn

# Helper for cleaner definitions
def ConvNormAct(in_ch, out_ch, k=3, s=1, p=1, g=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

OPERATIONS = {
    # --- Standard Baselines ---
    'identity': lambda in_ch, out_ch: nn.Sequential(nn.Identity()) if in_ch == out_ch else nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch)
    ),
    
    # --- Efficient Single Layers ---
    'conv_1x1': lambda in_ch, out_ch: ConvNormAct(in_ch, out_ch, k=1, p=0),
    
    'conv_3x3': lambda in_ch, out_ch: ConvNormAct(in_ch, out_ch, k=3, p=1),
    
    # --- CPU-Optimized Depthwise Separable ---
    # Strictly Depthwise (in->in) + Pointwise (in->out)
    'sep_conv_3x3': lambda in_ch, out_ch: nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    ),
    
    # --- High Capacity Blocks (U-Net Killers) ---
    # Standard Double Conv (Like U-Net, but with bias=False)
    'double_conv_3x3': lambda in_ch, out_ch: nn.Sequential(
        ConvNormAct(in_ch, out_ch, k=3),
        ConvNormAct(out_ch, out_ch, k=3)
    ),

    # --- High Capacity Blocks (U-Net Killers) ---
    # Standard Double Conv (Like U-Net, but with bias=False)
    'double_conv_3x3_d2': lambda in_ch, out_ch: nn.Sequential(
        ConvNormAct(in_ch, out_ch, k=3, d=2, p=2),
        ConvNormAct(out_ch, out_ch, k=3, d=2, p=2)
    ),

    'double_conv_3x3_d4': lambda in_ch, out_ch: nn.Sequential(
        ConvNormAct(in_ch, out_ch, k=3, d=4, p=4),
        ConvNormAct(out_ch, out_ch, k=3, d=4, p=4)
    ),


    # Residual Double Conv: Allows deep networks to train faster
    # Logic: Conv(in, out) -> ReLU -> Conv(out, out) + Skip
    'res_double_conv_3x3': lambda in_ch, out_ch: ResidualDoubleConv(in_ch, out_ch),

    # --- MobileNet Variants ---
    # MBConv with SE is slow on CPU. Provide a version WITHOUT SE.
    'mbconv_3x3_no_se': lambda in_ch, out_ch: MBConvBlock(
        in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, 
        expand_ratio=4, se_ratio=0.0 # Disabled SE
    ),
}

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        # Handle skip connection if channels mismatch
        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        return self.act(out)

# OPERATIONS = {
#     'conv_1x1': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'conv_1x3_3x1': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), padding=(0, 1), bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0), bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'depthwise_conv_3x3': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True, groups=math.gcd(in_ch, out_ch)),
#         nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'depthwise_conv_5x5': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, bias=True, groups=math.gcd(in_ch, out_ch)),
#         nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'conv_1x5_5x1': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=(1, 5), padding=(0, 2), bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'conv_3x3': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'double_conv_3x3': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'dilated_conv_3x3_r2': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, bias=True, dilation=2),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'dilated_conv_3x3_r4': lambda in_ch, out_ch: nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=4, bias=True, dilation=4),
#         nn.BatchNorm2d(out_ch),
#         nn.ReLU(inplace=True)
#     ),
#     'identity': lambda in_ch, out_ch: nn.Sequential(nn.Identity()) if in_ch == out_ch else nn.Sequential(
#         nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=True),
#         nn.BatchNorm2d(out_ch)
#     ), 
#     'mbconv_3x3': lambda in_ch, out_ch: MBConvBlock(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25),
# }

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.block(x)

# class SqueezeExcitation(nn.Module):
#     """
#     Squeeze-and-Excitation block for channel attention.
    
#     Adaptively recalibrates channel-wise feature responses by explicitly 
#     modeling interdependencies between channels.
#     """
    
#     def __init__(self, in_channels: int, squeeze_ratio: int = 4):
#         super().__init__()
#         squeezed_channels = max(1, in_channels // squeeze_ratio)
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, squeezed_channels, 1),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(squeezed_channels, in_channels, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.se(x)
    
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
                nn.Conv2d(in_channels, expanded_channels, 1, bias=True),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride=stride, padding=padding, groups=expanded_channels, bias=True
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
            nn.Conv2d(expanded_channels, out_channels, 1, bias=True),
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

# if __name__ == "__main__":
#     # Test the MBConvBlock
#     mbconv = MBConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25)
#     x = torch.randn(1, 32, 64, 64)
#     y = mbconv(x)
#     print(y.shape)  # Should be [1, 32, 64, 64]
#     # Count parameters
#     num_params = sum(p.numel() for p in mbconv.parameters())
#     print(f"Number of parameters: {num_params:,}")
    
#     # Test inference time on CPU
#     mbconv.eval()
#     for i in range(10): _ = mbconv(x)  # Warmup
#     with torch.no_grad():
#         start = time.time()
#         for _ in range(100):
#             _ = mbconv(x)
#         cpu_time = time.time() - start
#     print(f"CPU inference time (100 predictions): {cpu_time:.4f}s ({cpu_time/100*1000:.2f}ms per prediction)")
    
#     # Test inference time on GPU (if available)
#     if torch.cuda.is_available():
#         mbconv_gpu = mbconv.cuda()
#         x_gpu = x.cuda()
        
#         # Warmup
#         for _ in range(10):
#             _ = mbconv_gpu(x_gpu)
#         torch.cuda.synchronize()
        
#         start = time.time()
#         for _ in range(100):
#             _ = mbconv_gpu(x_gpu)
#         torch.cuda.synchronize()
#         gpu_time = time.time() - start
#         print(f"GPU inference time (100 predictions): {gpu_time:.4f}s ({gpu_time/100*1000:.2f}ms per prediction)")
#     else:
#         print("CUDA not available, skipping GPU test")

#     print("-=-=-=-=-==-=-=-=-=-=-=--=-=-=-=-=-=-=-==-=-=-=-==-=-")
#     mbconv = DoubleConv(in_channels=32, out_channels=32)
#     x = torch.randn(1, 32, 64, 64)
#     y = mbconv(x)
#     print(y.shape)  # Should be [1, 32, 64, 64]
#     # Count parameters
#     num_params = sum(p.numel() for p in mbconv.parameters())
#     print(f"Number of parameters: {num_params:,}")
    
#     # Test inference time on CPU
#     mbconv.eval()
#     for i in range(10): _ = mbconv(x)  # Warmup
#     with torch.no_grad():
#         start = time.time()
#         for _ in range(100):
#             _ = mbconv(x)
#         cpu_time = time.time() - start
#     print(f"CPU inference time (100 predictions): {cpu_time:.4f}s ({cpu_time/100*1000:.2f}ms per prediction)")
    
#     # Test inference time on GPU (if available)
#     if torch.cuda.is_available():
#         mbconv_gpu = mbconv.cuda()
#         x_gpu = x.cuda()
        
#         # Warmup
#         for _ in range(10):
#             _ = mbconv_gpu(x_gpu)
#         torch.cuda.synchronize()
        
#         start = time.time()
#         for _ in range(100):
#             _ = mbconv_gpu(x_gpu)
#         torch.cuda.synchronize()
#         gpu_time = time.time() - start
#         print(f"GPU inference time (100 predictions): {gpu_time:.4f}s ({gpu_time/100*1000:.2f}ms per prediction)")
#     else:
#         print("CUDA not available, skipping GPU test")