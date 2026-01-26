import tensorrt as trt
import os
import sys
import argparse

def build_engine(onnx_file_path, engine_file_path):
    # 1. Setup the Logger
    # TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # Use VERBOSE if you need to debug
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    print(f"Initializing TensorRT Builder...")
    builder = trt.Builder(TRT_LOGGER)
    
    # 2. Create the Network Definition (Explicit Batch is required for ONNX)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 3. Setup the ONNX Parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 4. Parse the ONNX file
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")
        
    print(f"Parsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 5. Configure the Builder
    config = builder.create_builder_config()
    
    # Enable FP16 (Half Precision) - Critical for speed on Tensor Cores
    if builder.platform_has_fast_fp16:
        print("Enabling FP16 precision...")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("Warning: FP16 not supported on this platform. Using FP32.")

    # Memory allocation (Allow up to 4GB for optimization workspace)
    # syntax depends on TRT version, this try-except handles both
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024) 
    except AttributeError:
        # Fallback for older TensorRT versions
        config.max_workspace_size = 4 * 1024 * 1024 * 1024

    # 6. Build the Serialized Engine
    print("Building engine... (This may take several minutes)")
    try:
        # Modern TRT versions
        serialized_engine = builder.build_serialized_network(network, config)
    except AttributeError:
        # Older TRT versions (7.x)
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize() if engine else None

    if serialized_engine is None:
        raise RuntimeError("Failed to build engine.")

    # 7. Save to disk
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"Success! Optimized engine saved to: {engine_file_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX model')
    parser.add_argument('--onnx', type=str, default='radar_network_sim.onnx',
                        help='Path to input ONNX file (default: radar_network_sim.onnx)')
    parser.add_argument('--output', type=str, default='radar_network.trt',
                        help='Path to output TensorRT engine file (default: radar_network.trt)')
    
    args = parser.parse_args()
    
    build_engine(args.onnx, args.output)