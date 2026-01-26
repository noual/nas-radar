import time
import numpy as np
import torch
import sys
import os

# Try importing optimization libraries
try:
    import onnxruntime as ort
except ImportError:
    ort = None
    print("Warning: onnxruntime not installed. ONNX benchmarks will be skipped.")

try:
    import tensorrt as trt
except ImportError:
    trt = None
    print("Warning: tensorrt not installed. TensorRT benchmarks will be skipped.")

class Benchmarker:
    def __init__(self, input_shape=(1, 1, 256, 256), n_warmup=10, n_runs=100):
        self.input_shape = input_shape
        self.n_warmup = n_warmup
        self.n_runs = n_runs
        self.results = {}
        
        # Create dummy data
        self.dummy_input_cpu = torch.randn(*input_shape)
        if torch.cuda.is_available():
            self.dummy_input_gpu = self.dummy_input_cpu.cuda()

    def _time_func(self, func, device_type="cpu"):
        """Generic timing loop with synchronization."""
        # Warmup
        for _ in range(self.n_warmup):
            func()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
            
        timings = []
        start_total = time.time()
        
        for _ in range(self.n_runs):
            t0 = time.time()
            func()
            # GPU sync is crucial for accurate measurement
            if device_type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.time() - t0) * 1000) # ms

        mean_time = np.mean(timings)
        std_time = np.std(timings)
        fps = 1000 / mean_time
        return mean_time, std_time, fps

    def benchmark_pytorch(self, model, name="PyTorch"):
        """Benchmark native PyTorch model."""
        print(f"\n--- Benchmarking {name} ---")
        model.eval()
        
        # 1. CPU
        try:
            model.cpu()
            with torch.no_grad():
                def run_cpu(): model(self.dummy_input_cpu)
                mu, std, fps = self._time_func(run_cpu, "cpu")
                self.results[f"{name} (CPU)"] = (mu, std, fps)
                print(f"CPU: {mu:.2f} ms ± {std:.2f} ms | {fps:.1f} FPS")
        except Exception as e:
            print(f"CPU Benchmark failed: {e}")

        # 2. GPU
        if torch.cuda.is_available():
            try:
                model.cuda()
                with torch.no_grad():
                    def run_gpu(): model(self.dummy_input_gpu)
                    mu, std, fps = self._time_func(run_gpu, "cuda")
                    self.results[f"{name} (GPU)"] = (mu, std, fps)
                    print(f"GPU: {mu:.2f} ms ± {std:.2f} ms | {fps:.1f} FPS")
            except Exception as e:
                print(f"GPU Benchmark failed: {e}")

    def benchmark_onnx(self, onnx_path):
        """Benchmark ONNX Runtime (CPU and GPU)."""
        if ort is None or not os.path.exists(onnx_path):
            return

        print(f"\n--- Benchmarking ONNX Runtime ---")
        numpy_input = self.dummy_input_cpu.numpy()

        # 1. CPU Provider
        try:
            sess_cpu = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_name = sess_cpu.get_inputs()[0].name
            
            def run_cpu():
                sess_cpu.run(None, {input_name: numpy_input})
                
            mu, std, fps = self._time_func(run_cpu, "cpu")
            self.results["ONNX (CPU)"] = (mu, std, fps)
            print(f"CPU: {mu:.2f} ms ± {std:.2f} ms | {fps:.1f} FPS")
        except Exception as e:
            print(f"ONNX CPU failed: {e}")

        # 2. GPU Provider
        if 'CUDAExecutionProvider' in ort.get_available_providers() and torch.cuda.is_available():
            try:
                sess_gpu = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
                input_name = sess_gpu.get_inputs()[0].name
                
                # Note: Passing numpy arrays to GPU provider forces a CPU->GPU copy.
                # Ideally we use IO Binding, but for simple benchmarking this is 'standard' usage.
                def run_gpu():
                    sess_gpu.run(None, {input_name: numpy_input})
                
                mu, std, fps = self._time_func(run_gpu, "cuda") # Sync happens inside run for ORT usually, but we sync anyway
                self.results["ONNX (GPU)"] = (mu, std, fps)
                print(f"GPU: {mu:.2f} ms ± {std:.2f} ms | {fps:.1f} FPS")
            except Exception as e:
                print(f"ONNX GPU failed: {e}")

    def benchmark_tensorrt(self, engine_path):
        """Benchmark TensorRT Engine."""
        if trt is None or not os.path.exists(engine_path):
            return
            
        print(f"\n--- Benchmarking TensorRT ---")
        
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # Allocate Inputs/Outputs using PyTorch (Zero copy, easy memory management)
        # We assume 1 input and 1 output for simplicity
        input_binding_idx = 0
        output_binding_idx = 1
        
        # Create output buffer
        # Note: You might need to query engine.get_binding_shape(1) if output shape differs
        output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
        # Handle dynamic batch dimension if present (usually -1)
        if output_shape[0] == -1:
            output_shape = (self.input_shape[0],) + output_shape[1:]
            
        output_tensor = torch.empty(output_shape, device='cuda', dtype=torch.float32)
        
        # Bindings are a list of pointers
        bindings = [0] * 2
        bindings[input_binding_idx] = self.dummy_input_gpu.data_ptr()
        bindings[output_binding_idx] = output_tensor.data_ptr()
        
        def run_trt():
            # execute_v2 is for explicit batch (standard for ONNX parsed models)
            context.execute_v2(bindings=bindings)
            
        mu, std, fps = self._time_func(run_trt, "cuda")
        self.results["TensorRT (GPU)"] = (mu, std, fps)
        print(f"GPU: {mu:.2f} ms ± {std:.2f} ms | {fps:.1f} FPS")

    def print_summary(self):
        print("\n" + "="*50)
        print(f"{'Framework':<20} | {'Device':<5} | {'Latency (ms)':<15} | {'FPS':<10}")
        print("-" * 50)
        
        # Sort by FPS descending
        sorted_res = sorted(self.results.items(), key=lambda x: x[1][2], reverse=True)
        
        for name, (mu, std, fps) in sorted_res:
            dev = "GPU" if "GPU" in name else "CPU"
            clean_name = name.replace(" (GPU)", "").replace(" (CPU)", "")
            print(f"{clean_name:<20} | {dev:<5} | {mu:6.2f} ± {std:4.2f} | {fps:6.1f}")
        print("="*50)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # 1. Define your input shape
    INPUT_SHAPE = (1, 1, 128, 128) # (B, C, H, W)
    
    # 2. Define File Paths
    ONNX_PATH = "best_nrpa_model_sim.onnx"
    TRT_PATH = "best_nrpa_model.trt"
    
    # 3. Setup Benchmarker
    bm = Benchmarker(input_shape=INPUT_SHAPE, n_warmup=50, n_runs=200)
    
    # --- LOAD YOUR PYTORCH MODEL HERE ---
    # For demonstration, we use a dummy wrapper. 
    # REPLACE THIS with: model = supernet.sample(node); model.prune_cells()
    print("Loading PyTorch model...")
    # Example placeholder:
    # model = ... (Load your model as you did in training)
    # bm.benchmark_pytorch(model, name="NAS-Model")
    
    # --- RUN BENCHMARKS ---
    bm.benchmark_onnx(ONNX_PATH)
    bm.benchmark_tensorrt(TRT_PATH)
    
    bm.print_summary()