import torch
import onnx
import onnxsim

import sys
sys.path.append("..")

def optimize_and_export_to_onnx(model, file_path):

    # 1. Setup your model (Same as your training script)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # 3. Export to ONNX
    # Use a static batch size (e.g., 1) for maximum speed optimization
    dummy_input = torch.randn(1, 1, 128, 128, device=device) 
    onnx_path = file_path

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,  # Fuses constants
        input_names=['input'],
        output_names=['output'],
        # Removing dynamic axes allows the compiler to fully unroll loops
        dynamic_axes=None 
    )

    # 4. Simplify the Graph
    # NAS models often generate redundant "Identity" or "Transpose" nodes. 
    # onnx-simplifier removes them.
    print("Simplifying ONNX graph...")
    model_onnx = onnx.load(onnx_path)
    model_simp, check = onnxsim.simplify(model_onnx)

    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, file_path.replace(".onnx", "_sim.onnx"))
    print(f"Optimization complete. Saved to '{file_path.replace('.onnx', '_sim.onnx')}'")