import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block replacing standard convolution.
    
    Consists of:
    1. Depthwise convolution (spatial filtering)
    2. Pointwise convolution (channel mixing)
    3. BatchNorm and activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EfficientDoubleConv(nn.Module):
    """
    Efficient version of DoubleConv using depthwise separable convolutions.
    Can be made deeper while maintaining similar parameter count.
    """
    def __init__(self, in_channels, out_channels, num_layers=3):
        super().__init__()
        
        # First layer: in_channels -> out_channels
        layers = [DepthwiseSeparableConv(in_channels, out_channels)]
        
        # Additional layers: out_channels -> out_channels (makes it deeper)
        for _ in range(num_layers - 1):
            layers.append(DepthwiseSeparableConv(out_channels, out_channels))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class EfficientUNet(nn.Module):
    """
    Efficient UNet using depthwise separable convolutions.
    
    Key improvements:
    - Depthwise separable convs instead of standard convs
    - Deeper blocks (3 layers vs 2) for better feature extraction
    - Significantly fewer parameters
    - Better parameter efficiency
    """
    def __init__(self, in_channels=1, initial_channels=8, out_channels=1, 
                 features=[16, 32, 64, 128], num_layers_per_block=3):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        # Initial convolution to map input to initial_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        in_channels = initial_channels

        # Encoder with deeper efficient blocks
        for feature in features:
            self.downs.append(EfficientDoubleConv(in_channels, feature, num_layers_per_block))
            in_channels = feature

        # Bottleneck - extra deep for better representation
        self.bottleneck = EfficientDoubleConv(features[-1], features[-1]*2, num_layers_per_block + 1)

        # Decoder
        for feature in reversed(features):
            # Upsample + reduce channels
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                DepthwiseSeparableConv(feature*2, feature)
            ))
            # Process concatenated features
            self.ups.append(EfficientDoubleConv(feature*2, feature, num_layers_per_block))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        skip_connections = []
        
        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            # Upsample
            x = self.ups[idx](x)
            skip_conn = skip_connections[idx//2]

            # Handle size mismatch
            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:])

            # Concatenate and process
            x = torch.cat((skip_conn, x), dim=1)
            x = self.ups[idx+1](x)
        
        return self.final_conv(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def benchmark_models():
    """Compare EfficientUNet vs standard UNet"""
    
    # Import the original UNet
    from unet import UNet
    
    # Create models with similar configurations
    features = [64, 128, 256, 512]
    device_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Standard UNet
    unet = UNet(in_channels=1, initial_channels=8, out_channels=1, features=features)
    
    # Efficient UNet - deeper but fewer params
    efficient_unet = EfficientUNet(
        in_channels=1, initial_channels=8, out_channels=1, 
        features=features, num_layers_per_block=3
    )
    
    print("=== Model Comparison ===")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"EfficientUNet parameters: {efficient_unet.count_parameters():,}")
    print(f"Parameter ratio: {sum(p.numel() for p in unet.parameters()) / efficient_unet.count_parameters():.2f}x")
    
    # Test input
    batch_size = 2
    height, width = 256, 256
    x_cpu = torch.randn(batch_size, 1, height, width)
    
    # ==================== GPU BENCHMARK ====================
    if torch.cuda.is_available():
        print(f"\n=== GPU Benchmark ({device_gpu}) ===")
        
        unet_gpu = unet.to(device_gpu).eval()
        efficient_unet_gpu = efficient_unet.to(device_gpu).eval()
        x_gpu = x_cpu.to(device_gpu)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = unet_gpu(x_gpu)
                _ = efficient_unet_gpu(x_gpu)
            torch.cuda.synchronize()
        
        # Benchmark UNet
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                _ = unet_gpu(x_gpu)
            torch.cuda.synchronize()
            unet_gpu_time = (time.time() - t0) / 50
        
        # Benchmark EfficientUNet
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                _ = efficient_unet_gpu(x_gpu)
            torch.cuda.synchronize()
            efficient_gpu_time = (time.time() - t0) / 50
        
        print(f"UNet GPU time: {unet_gpu_time*1000:.3f} ms")
        print(f"EfficientUNet GPU time: {efficient_gpu_time*1000:.3f} ms")
        print(f"GPU speedup: {unet_gpu_time/efficient_gpu_time:.2f}x ({'EfficientUNet' if efficient_gpu_time < unet_gpu_time else 'UNet'} is faster)")
        
        # Test output shapes
        with torch.no_grad():
            out_unet = unet_gpu(x_gpu)
            out_efficient = efficient_unet_gpu(x_gpu)
        print(f"UNet output shape: {out_unet.shape}")
        print(f"EfficientUNet output shape: {out_efficient.shape}")
    
    # ==================== CPU BENCHMARK ====================
    print(f"\n=== CPU Benchmark ===")
    
    unet_cpu = unet.to("cpu").eval()
    efficient_unet_cpu = efficient_unet.to("cpu").eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = unet_cpu(x_cpu)
            _ = efficient_unet_cpu(x_cpu)
    
    # Benchmark UNet on CPU
    with torch.no_grad():
        t0 = time.time()
        for _ in range(10):
            _ = unet_cpu(x_cpu)
        unet_cpu_time = (time.time() - t0) / 10
    
    # Benchmark EfficientUNet on CPU
    with torch.no_grad():
        t0 = time.time()
        for _ in range(10):
            _ = efficient_unet_cpu(x_cpu)
        efficient_cpu_time = (time.time() - t0) / 10
    
    print(f"UNet CPU time: {unet_cpu_time*1000:.3f} ms")
    print(f"EfficientUNet CPU time: {efficient_cpu_time*1000:.3f} ms")
    print(f"CPU speedup: {unet_cpu_time/efficient_cpu_time:.2f}x ({'EfficientUNet' if efficient_cpu_time < unet_cpu_time else 'UNet'} is faster)")
    
    # ==================== MEMORY ANALYSIS ====================
    print(f"\n=== Memory Analysis ===")
    
    def get_model_memory(model, x):
        """Estimate model memory usage"""
        model_params = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        # Rough activation memory estimate (forward pass)
        with torch.no_grad():
            try:
                if x.is_cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    _ = model(x)
                    if x.is_cuda:
                        activation_mem = torch.cuda.max_memory_allocated() / (1024**2)
                    else:
                        activation_mem = 0  # Hard to measure on CPU
                else:
                    activation_mem = 0
            except Exception:
                activation_mem = 0
        
        return model_params, activation_mem
    
    if torch.cuda.is_available():
        unet_params_mem, unet_activation_mem = get_model_memory(unet_gpu, x_gpu)
        eff_params_mem, eff_activation_mem = get_model_memory(efficient_unet_gpu, x_gpu)
        
        print(f"UNet - Parameters: {unet_params_mem:.2f} MB, Peak activation: {unet_activation_mem:.2f} MB")
        print(f"EfficientUNet - Parameters: {eff_params_mem:.2f} MB, Peak activation: {eff_activation_mem:.2f} MB")
        print(f"Memory savings: {(unet_params_mem + unet_activation_mem) / (eff_params_mem + eff_activation_mem):.2f}x")
    
    # ==================== SUMMARY ====================
    print(f"\n=== Summary ===")
    print(f"✓ EfficientUNet has {efficient_unet.count_parameters()/sum(p.numel() for p in unet.parameters())*100:.1f}% of UNet parameters")
    print(f"✓ EfficientUNet is deeper ({3} layers per block vs {2})")
    if torch.cuda.is_available():
        print(f"✓ GPU: {'EfficientUNet' if efficient_gpu_time < unet_gpu_time else 'UNet'} is {abs(unet_gpu_time/efficient_gpu_time if efficient_gpu_time < unet_gpu_time else efficient_gpu_time/unet_gpu_time):.2f}x faster")
    print(f"✓ CPU: {'EfficientUNet' if efficient_cpu_time < unet_cpu_time else 'UNet'} is {abs(unet_cpu_time/efficient_cpu_time if efficient_cpu_time < unet_cpu_time else efficient_cpu_time/unet_cpu_time):.2f}x faster")


if __name__ == "__main__":
    benchmark_models()