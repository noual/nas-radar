import torch
from torch.func import functional_call, vmap, jacrev

def compute_ntk(model, x, chunk_size=1, use_fp16=False):
    """
    Compute NTK matrix with memory-efficient chunking.
    
    The NTK is computed as K(x_i, x_j) = (1/d) * sum_k (df_k/dθ)(x_i) · (df_k/dθ)(x_j)
    where d is the output dimension. This normalization ensures O(1) NTK values
    regardless of the output size, which is important for segmentation networks
    with large spatial outputs.
    
    Args:
        model: Neural network model
        x: Input batch of shape (N, C, H, W)
        chunk_size: Number of samples to process at once (default=1 for minimal memory)
        use_fp16: Currently disabled due to functional_call limitations with BatchNorm.
                  All computations are done in FP32 for numerical stability.
    
    Returns:
        NTK matrix of shape (N, N) in FP32, normalized by output dimension
    """
    # Initialize weights for NTK computation following NTK theory
    # Use standard Gaussian initialization (mean=0, std=1) without fan-in scaling
    # This is critical for NTK regime where we want O(1) kernel values
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            # NTK initialization: N(0, 1/fan_in) for proper scaling
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0/m.in_features**0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv2d):
            # NTK initialization for conv layers
            fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0/fan_in**0.5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            # For NTK computation, set BatchNorm to "pass-through" mode
            # Set weight to 1 and bias to 0, then freeze
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
            # Set running stats to identity transformation
            if hasattr(m, 'running_mean'):
                m.running_mean.zero_()
            if hasattr(m, 'running_var'):
                m.running_var.fill_(1.0)
    
    model.apply(init_weights)
    model.eval()  # Disable dropout
    
    n_samples = x.shape[0]
    # NTK accumulation always in FP32 for numerical stability
    ntk = torch.zeros(n_samples, n_samples, device=x.device, dtype=torch.float32)
    
    # Note: FP16 is problematic with functional_call and BatchNorm
    # We'll convert the model but keep computations in FP32 for stability
    if use_fp16:
        # Convert model to FP16 but this may cause issues with BatchNorm
        # Better approach: just use FP16 for storage, compute in FP32
        x = x.half()
        # Don't convert model - keep in FP32 for functional_call compatibility
        use_fp16 = False  # Force FP32 for now due to functional_call limitations
    
    # Get parameters - only include parameters that require gradients
    parameters = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    
    # Get output dimension for normalization
    with torch.no_grad():
        sample_output = model(x[0:1])
        output_dim = sample_output.numel()
    
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
    
    # Normalize by output dimension to get O(1) NTK values
    # This is critical for comparing networks with different output sizes
    ntk = ntk / output_dim
    
    return ntk

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")
    from search_spaces.radar_node import RadarNode

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

    eigenvalues = torch.linalg.eigvalsh(ntk_fp32)
    min_eigenvalue = eigenvalues[0]
    trace_norm = torch.trace(ntk_fp32)

    print(f"\nMinimum eigenvalue (FP16): {min_eigenvalue.item():.6f}")
    print(f"Trace norm (FP16): {trace_norm.item():.6f}")