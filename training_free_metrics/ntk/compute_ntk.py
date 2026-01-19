import torch
from torch.func import functional_call, vmap, jacrev

def compute_ntk(model, x, chunk_size=1, use_fp16=False):
    """
    Compute NTK matrix with memory-efficient chunking.
    
    Args:
        model: Neural network model
        x: Input batch of shape (N, C, H, W)
        chunk_size: Number of samples to process at once (default=1 for minimal memory)
        use_fp16: If True, use FP16 for forward passes and Jacobians (saves memory, faster on modern GPUs)
                  NTK accumulation is done in FP32 for numerical stability
    
    Returns:
        NTK matrix of shape (N, N) in FP32
    """
    model.eval()  # Disable dropout
    
    n_samples = x.shape[0]
    # NTK accumulation always in FP32 for numerical stability
    ntk = torch.zeros(n_samples, n_samples, device=x.device, dtype=torch.float32)
    
    # Convert input to FP16 if requested
    if use_fp16:
        x = x.half()
    
    parameters = {k: v.detach().half() if use_fp16 else v.detach() 
                  for k, v in model.named_parameters()}
    
    def fnet_single(params, x_single):
        x_single = x_single.unsqueeze(0)
        output = functional_call(model, params, x_single)
        return output.squeeze(0).flatten()
    
    # Process in chunks to reduce memory usage
    for i in range(0, n_samples, chunk_size):
        i_end = min(i + chunk_size, n_samples)
        x_chunk_i = x[i:i_end]
        
        # Compute Jacobian for chunk i
        jac_i = vmap(jacrev(fnet_single), (None, 0), randomness='same')(parameters, x_chunk_i)
        
        for j in range(0, n_samples, chunk_size):
            j_end = min(j + chunk_size, n_samples)
            x_chunk_j = x[j:j_end]
            
            # Compute Jacobian for chunk j
            jac_j = vmap(jacrev(fnet_single), (None, 0), randomness='same')(parameters, x_chunk_j)
            
            # Compute NTK block: J_i @ J_j^T, processing per parameter to save memory
            # Accumulate in FP32 for numerical stability
            ntk_block = torch.zeros(i_end - i, j_end - j, device=x.device, dtype=torch.float32)
            for param_name in jac_i.keys():
                j_i = jac_i[param_name].flatten(1)  # (chunk_i, n_params)
                j_j = jac_j[param_name].flatten(1)  # (chunk_j, n_params)
                # Convert to FP32 for accumulation if using FP16
                if use_fp16:
                    ntk_block += torch.mm(j_i.float(), j_j.float().t())
                else:
                    ntk_block += torch.mm(j_i, j_j.t())
            
            ntk[i:i_end, j:j_end] = ntk_block
            
            # Clear jacobian j to free memory
            del jac_j
            torch.cuda.empty_cache() if x.is_cuda else None
        
        # Clear jacobian i to free memory
        del jac_i
        torch.cuda.empty_cache() if x.is_cuda else None
    
    return ntk

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")
    from search_space.radar_node import RadarNode

    node = RadarNode(in_channels=3, initial_channels=8, channel_options=[8, 16, 32, 64], num_encoder_stages=3, num_nodes=4)
    while not node.is_terminal:
        actions = node.get_actions_tuples()
        action = actions[0]  # Just pick the first action for testing
        node.play_action(action)
    
    model = node.network
    model.prune_cells()
    model.to("cuda:0")
    x = torch.randn(10, 3, 32, 32).to("cuda:0")  # Example input batch of size 10
    
    # Compare FP32 vs FP16
    print("Computing NTK with FP32...")
    ntk_fp32 = compute_ntk(model, x, chunk_size=1, use_fp16=False)
    print("NTK matrix shape:", ntk_fp32.shape)
    print("NTK condition number (FP32):", torch.linalg.cond(ntk_fp32).item())
    
    print("\nComputing NTK with FP16...")
    ntk_fp16 = compute_ntk(model, x, chunk_size=1, use_fp16=True)
    print("NTK condition number (FP16):", torch.linalg.cond(ntk_fp16).item())
    
    # Check numerical difference
    rel_error = torch.norm(ntk_fp32 - ntk_fp16) / torch.norm(ntk_fp32)
    print(f"Relative error between FP32 and FP16: {rel_error.item():.6f}")