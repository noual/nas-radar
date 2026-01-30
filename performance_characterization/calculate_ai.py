import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys

sys.path.append('.')
sys.path.append('..')
from utils.models.efficient_unet import EfficientUNet
from utils.models.unet import UNet

class ArithmeticIntensityTracker:
    def __init__(self, model, input_size, batch_size=1, dtype=torch.float32):
        """
        Args:
            model: The PyTorch model to profile.
            input_size: Tuple of input shape (e.g., (3, 224, 224)).
            batch_size: Batch size for profiling (default 1 for latency/inference).
            dtype: Data type (default float32 = 4 bytes).
        """
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.dtype_size = 4 if dtype == torch.float32 else 2 # 4 bytes for fp32
        self.layer_stats = []
        self.hooks = []
        
        # CPU Ridge Point (approximate for modern CPUs like Xeon Scalable / Core i9)
        # Adjust this based on your specific hardware (e.g., 15.0 for AVX512 systems)
        self.TARGET_RIDGE_POINT = 6.6

    def _register_hooks(self):
        """Registers hooks to leaf modules."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.SiLU, nn.MaxPool2d, nn.AvgPool2d)):
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
            
            # If you have custom modules (like 'PeakConv'), add them here:
            # if "PeakConv" in str(type(module)):...

    def _hook_fn(self, name):
        def hook(module, input, output):
            # Get input/output shapes
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            
            flops = 0
            mem_bytes = 0
            b_sz = inp.size(0)
            
            # --- LOGIC FOR CONVOLUTION ---
            if isinstance(module, nn.Conv2d):
                # FLOPs: 2 * Cin * K * K * Hout * Wout * Cout / groups
                # Note: We divide by groups for Depthwise Conv
                k_h, k_w = module.kernel_size
                in_c = module.in_channels
                out_c = module.out_channels
                h_out, w_out = output.size(2), output.size(3)
                groups = module.groups
                
                # Standard MACs * 2 (for Add+Mul)
                flops = 2 * (in_c // groups) * k_h * k_w * h_out * w_out * out_c * b_sz
                
                if module.bias is not None:
                    flops += out_c * h_out * w_out * b_sz
                
                # Memory: Read Input + Read Weights + Write Output
                # We assume DRAM access for all (worst case / Roofline P1)
                input_bytes = inp.numel() * self.dtype_size
                output_bytes = output.numel() * self.dtype_size
                weight_bytes = module.weight.numel() * self.dtype_size
                
                if module.bias is not None:
                    weight_bytes += module.bias.numel() * self.dtype_size
                
                mem_bytes = input_bytes + weight_bytes + output_bytes

            # --- LOGIC FOR LINEAR ---
            elif isinstance(module, nn.Linear):
                in_f = module.in_features
                out_f = module.out_features
                
                flops = 2 * in_f * out_f * b_sz
                if module.bias is not None:
                    flops += out_f * b_sz
                    
                input_bytes = inp.numel() * self.dtype_size
                output_bytes = output.numel() * self.dtype_size
                weight_bytes = module.weight.numel() * self.dtype_size
                
                mem_bytes = input_bytes + weight_bytes + output_bytes

            # --- LOGIC FOR FUSED OPS (BN, ReLU) ---
            elif isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.SiLU)):
                # ASSUMPTION: These are FUSED into the preceding Conv on CPU.
                # Therefore, they incur ZERO extra DRAM traffic and negligible FLOPs 
                # relative to Conv. We track them but assume 0 bytes to reward fusion.
                flops = inp.numel() * b_sz # 1 op per element
                mem_bytes = 0 # Fused!

            # --- LOGIC FOR POOLING ---
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                # Pooling is purely memory bound.
                flops = inp.numel() * b_sz
                mem_bytes = (inp.numel() + output.numel()) * self.dtype_size

            # Store stats
            ai = flops / mem_bytes if mem_bytes > 0 else 0
            self.layer_stats.append({
                'name': name,
                'type': type(module).__name__,
                'flops': flops,
                'bytes': mem_bytes,
                'ai': ai,
                'shape': str(list(output.shape))
            })
            
        return hook

    def calculate(self, verbose=False):
        # 1. Register hooks
        self._register_hooks()
        
        # 2. Run dummy forward pass
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(self.batch_size, *self.input_size).to(device)
        
        with torch.no_grad():
            self.model(dummy_input)
            
        # 3. Clean up hooks
        for h in self.hooks:
            h.remove()
            
        # 4. Aggregate Results
        df = pd.DataFrame(self.layer_stats)
        
        # Calculate Global AI
        total_flops = df['flops'].sum()
        total_bytes = df['bytes'].sum()
        global_ai = total_flops / total_bytes if total_bytes > 0 else 0
        if verbose:
            # 5. Analysis
            print(f"\n{'='*20} ARITHMETIC INTENSITY REPORT {'='*20}")
            print(f"Total FLOPs: {total_flops / 1e6:.2f} MFLOPs")
            print(f"Total Data:  {total_bytes / 1e6:.2f} MB")
            print(f"Global AI:   {global_ai:.2f} FLOPs/Byte")
            print(f"-"*60)
            
            if global_ai < self.TARGET_RIDGE_POINT:
                print(f"⚠️  STATUS: MEMORY BOUND (Zone P1)")
                print(f"   The model is starving the CPU. It waits for data {self.TARGET_RIDGE_POINT/global_ai:.1f}x longer than it computes.")
                print(f"   ACTION: Increase channel widths or replace Depthwise Convs with Standard Convs.")
            else:
                print(f"✅  STATUS: COMPUTE BOUND (Zone P2)")
                print(f"   The model is utilizing CPU vector units efficiently.")
                
            print(f"\n{'='*20} LAYER-WISE BREAKDOWN {'='*20}")
            # Show top 5 lowest AI layers (the bottlenecks)
            df_nn = df[df['ai'] > 0] # Ignore 0 AI layers (fused)
            print(df_nn.sort_values(by='ai', ascending=True)[['name', 'type', 'ai', 'flops']].head(10))
        
        return global_ai, df
    
def calculate_ridge_point(cores, frequency_ghz, memory_bw_gb_s, instruction_set='AVX512'):
    """
    Calculates the Theoretical Ridge Point for a CPU.
    """
    # 1. Determine FLOPs per Cycle per Core
    if instruction_set == 'AVX512':
        # 512-bit vector = 16 FP32 elements
        # Most Client CPUs (Ice Lake/Tiger Lake/Rocket Lake) have 1 FMA unit.
        # Server CPUs (Xeon Gold/Platinum) often have 2 FMA units.
        vec_width = 16
        fma_units = 1 # Assume Client CPU (Change to 2 for Server)
        ops_per_fma = 2 # Mul + Add
        flops_per_cycle = vec_width * fma_units * ops_per_fma
        
    elif instruction_set == 'AVX2':
        # 256-bit vector = 8 FP32 elements
        # Most modern CPUs have 2 FMA units for AVX2.
        vec_width = 8
        fma_units = 2
        ops_per_fma = 2
        flops_per_cycle = vec_width * fma_units * ops_per_fma
        
    else: # NEON (ARM / M1 / M2)
        # 128-bit vector = 4 FP32 elements
        # M1 has 4 SIMD units
        flops_per_cycle = 4 * 4 * 2 # Approx 32

    # 2. Calculate Peak GFLOPs
    peak_gflops = cores * frequency_ghz * flops_per_cycle
    
    # 3. Calculate Ridge Point
    ridge_point = peak_gflops / memory_bw_gb_s
    
    return peak_gflops, ridge_point


# --- Usage Example ---
if __name__ == "__main__":
    # --- EXAMPLE FOR I7-1065G7 (Ice Lake) ---
    gflops, ridge = calculate_ridge_point(
        cores=8, 
        frequency_ghz=5,     # Est. All-Core Turbo
        memory_bw_gb_s=58.3,   # LPDDR4x-3733
        instruction_set='AVX512'
    )

    print(f"Target CPU (Ice Lake i7):")
    print(f"  Peak Compute: {gflops:.2f} GFLOPs/s")
    print(f"  Ridge Point:  {ridge:.2f} FLOPs/Byte")
    print(f"  (Layers with AI < {ridge:.1f} will stall the CPU)")

    # 1. Define a Mock Frugal Network (Depthwise Separable)
    class FrugalBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            # Depthwise (Groups=C) -> Low AI
            self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False) 
            self.pw = nn.Conv2d(c, c, 1, bias=False)
            self.bn = nn.BatchNorm2d(c)
            self.act = nn.ReLU()
            
        def forward(self, x):
            return self.act(self.bn(self.pw(self.dw(x))))

    # 2. Define a U-Net Style Block (Dense)
    class UNetBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            # Dense 3x3 -> High AI
            self.conv = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(c)
            self.act = nn.ReLU()
            self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(c)
            self.act2 = nn.ReLU()
            
        def forward(self, x):
            x = self.act(self.bn(self.conv(x)))
            x = self.act2(self.bn2(self.conv2(x)))
            return x

    print(">>> ANALYZING FRUGAL BLOCK (Depthwise)")
    frugal_net = FrugalBlock(8)
    tracker = ArithmeticIntensityTracker(frugal_net, (8, 64, 64)) # Input: 8ch, 64x64
    tracker.calculate()

    print("\n\n>>> ANALYZING U-NET BLOCK (Standard)")
    unet_net = UNetBlock(8)
    tracker = ArithmeticIntensityTracker(unet_net, (8, 64, 64))
    tracker.calculate()

    unet = UNet(in_channels=3, out_channels=1, features=[16, 32, 64])
    print("\n\n>>> ANALYZING FULL U-NET MODEL")
    tracker = ArithmeticIntensityTracker(unet, (3, 128, 128))
    tracker.calculate()

    mod = EfficientUNet(in_channels=3, out_channels=1, features=[16, 32, 64])
    print("\n\n>>> ANALYZING FULL EFFICIENT U-NET MODEL")
    tracker = ArithmeticIntensityTracker(mod, (3, 128, 128))
    tracker.calculate()