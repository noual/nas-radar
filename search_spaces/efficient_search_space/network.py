import copy
import itertools
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(".")
sys.path.append("..")

from search_spaces.efficient_search_space.operations import OPERATIONS

class MaximalFrugalRadarNetwork(nn.Module):
    
    def __init__(self, in_channels=3, initial_channels=8, max_channels=512, num_encoder_stages=3, device="cpu"):
        super().__init__()
        self.out_channels = in_channels
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.ModuleDict()
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.in_channels = in_channels
        self.initial_channels = initial_channels
        self.max_channels = max_channels
        self.device = device

        # Initial convolution to map input to initial_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        in_channels = initial_channels

        # Encoder Construction
        for i in range(num_encoder_stages):
            out_channels = max_channels
            module_dict = nn.ModuleDict()
            for op_name, operation in OPERATIONS.items():
                module_dict[op_name] = operation(in_channels, out_channels)
            self.encoder.append(module_dict)
            in_channels = out_channels

        # Bottleneck Construction
        for op_name, operation in OPERATIONS.items():
            self.bottleneck[op_name] = operation(in_channels, out_channels)
        in_channels = out_channels
        
        # Decoder Construction
        for i in range(num_encoder_stages):
            out_channels = max_channels
            module_dict = nn.ModuleDict()
            
            # CRITICAL FIX: Decoder inputs include skip connections. 
            # Max input = (Max Prev Decoder) + (Max Skip) = 512 + 512 = 1024
            max_input_channels = self.max_channels * 2
            
            for op_name, operation in OPERATIONS.items():
                module_dict[op_name] = operation(max_input_channels, out_channels)
                
            self.decoder.append(module_dict)
            in_channels = out_channels

        self.final_conv = nn.Conv2d(max_channels, 1, kernel_size=1)
        self.to(device)

    def forward(self, x, selected_ops, channel_config):
        x = self.stem(x)
        skip_connections = []
        
        # Encoder
        for (i, module_dict), channels in zip(enumerate(self.encoder), channel_config['encoder']):
            op_name = selected_ops['encoder'][i]
            x = apply_sliced_operation(module_dict[op_name], x, in_channels=x.size(1), out_channels=channels)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        bottleneck_op = selected_ops['bottleneck']
        bottleneck_channels = channel_config['bottleneck']
        x = apply_sliced_operation(self.bottleneck[bottleneck_op], x, in_channels=x.size(1), out_channels=bottleneck_channels)
        
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for (i, module_dict), channels in zip(enumerate(self.decoder), channel_config['decoder']):
            op_name = selected_ops['decoder'][i]
            x = self.up(x)
            skip_conn = skip_connections[i]
            
            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:])
            
            x = torch.cat((skip_conn, x), dim=1)
            
            # Input channels to operation is current x channels (skip + prev_out)
            x = apply_sliced_operation(module_dict[op_name], x, in_channels=x.size(1), out_channels=channels)

        sliced_weight = self.final_conv.weight[:self.out_channels, :x.size(1), :, :]
        sliced_bias = self.final_conv.bias[:self.out_channels] if self.final_conv.bias is not None else None
        
        x = F.conv2d(x, sliced_weight, sliced_bias, 
                    self.final_conv.stride, self.final_conv.padding, 
                    self.final_conv.dilation, self.final_conv.groups)
        return x

class FrugalRadarNetwork(nn.Module):
    # (Unchanged from your provided code, omitted for brevity but required for full file)
    def __init__(self, in_channels=3, initial_channels=8, out_channels=1, num_encoder_stages=2, device="cpu"):
        super().__init__()
        self.out_channels = out_channels
        self.channel_list = {"encoder": [None for e in range(num_encoder_stages)], "bottleneck": None, "decoder": [None for e in range(num_encoder_stages)]}
        self.encoder = nn.ModuleList()
        self.bottleneck = None
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        self.final_conv = None
        self.device = device

    def play_action(self, action):
        if not action[0].endswith("channels"):
            if action[0].startswith("encoder"):
                stage_idx = int(action[0].split("_")[1])
                if self.channel_list["encoder"][stage_idx] is None:
                    raise ValueError(f"Encoder stage {stage_idx} channels must be set before setting operation.")
                op_name = action[1]
                in_ch = self.stem[0].out_channels if stage_idx == 0 else self.channel_list["encoder"][stage_idx - 1]
                out_ch = self.channel_list["encoder"][stage_idx]
                operation = OPERATIONS[op_name](in_ch, out_ch)
                self.encoder.append(operation)
            elif action[0].startswith("bottleneck"):
                op_name = action[1]
                if self.channel_list["bottleneck"] is None:
                    raise ValueError(f"Bottleneck channels must be set before setting operation.")
                in_ch = self.channel_list["encoder"][-1]
                out_ch = self.channel_list["bottleneck"]
                operation = OPERATIONS[op_name](in_ch, out_ch)
                self.bottleneck = operation
            elif action[0].startswith("decoder"):  
                stage_idx = int(action[0].split("_")[1])
                if self.channel_list["decoder"][stage_idx] is None:
                    raise ValueError(f"Decoder stage {stage_idx} channels must be set before setting operation.")
                
                op_name = action[1]
                in_ch = self.channel_list["bottleneck"] if stage_idx == 0 else self.channel_list["decoder"][stage_idx - 1]
                in_ch += self.channel_list["encoder"][-(stage_idx + 1)] 
                out_ch = self.channel_list["decoder"][stage_idx]
                operation = OPERATIONS[op_name](in_ch, out_ch)
                self.decoder.append(operation)
        else:
            if action[0].startswith("encoder"):
                stage_idx = int(action[0].split("_")[1])
                self.channel_list["encoder"][stage_idx] = action[1]
            elif action[0].startswith("bottleneck"):
                self.channel_list["bottleneck"] = action[1]
            elif action[0].startswith("decoder"):
                stage_idx = int(action[0].split("_")[1])
                self.channel_list["decoder"][stage_idx] = action[1]

    @property
    def is_terminal(self):
        if self.bottleneck is None:
            return False
        for module in itertools.chain(self.encoder, self.decoder):
            if module is None:
                return False
        if self.final_conv is None:
            self.final_conv = nn.Conv2d(self.channel_list["decoder"][-1], 1, kernel_size=1, device=self.device)
            self.to(device=self.device)
        return True

    def forward(self, x):
        assert self.is_terminal, "Network is not fully defined."
        x = self.stem(x)
        skip_connections = []
        for i, module in enumerate(self.encoder):
            x = module(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i, module in enumerate(self.decoder):
            x = self.up(x)
            skip_conn = skip_connections[i]
            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:])
            x = torch.cat((skip_conn, x), dim=1)
            x = module(x)

        x = self.final_conv(x)
        return x

# --- SLICING LOGIC ---

def apply_sliced_operation(module, x, in_channels, out_channels):
    # 1. Base Primitives
    if isinstance(module, nn.Identity):
        if in_channels == out_channels:
            return x
        elif in_channels < out_channels:
            return F.pad(x, (0, 0, 0, 0, 0, out_channels - in_channels))
        else:
            return x[:, :out_channels, :, :]

    if isinstance(module, nn.Conv2d):
        return _apply_sliced_conv(module, x, out_channels)

    if isinstance(module, nn.BatchNorm2d):
        return _apply_sliced_bn(module, x)

    if isinstance(module, (nn.ReLU, nn.SiLU, nn.Sigmoid, nn.AdaptiveAvgPool2d)):
        return module(x)

    # 2. Complex Blocks
    class_name = module.__class__.__name__

    if "MBConvBlock" in class_name:
        return _apply_sliced_mbconv(module, x, out_channels)
    
    if "SqueezeExcitation" in class_name:
        return _apply_sliced_se(module, x)

    if "DoubleConv" in class_name and hasattr(module, 'block'):
        return apply_sliced_operation(module.block, x, in_channels, out_channels)

    if "ResidualDoubleConv" in class_name:
        return _apply_sliced_res_double(module, x, out_channels)

    # 3. Containers
    if isinstance(module, (nn.Sequential, nn.ModuleList, list)):
        return _apply_sliced_sequential(module, x, out_channels)

    return module(x)


def _apply_sliced_res_double(module, x, final_out_channels):
    out = _apply_sliced_conv(module.conv1, x, final_out_channels)
    out = _apply_sliced_bn(module.bn1, out)
    out = module.act(out)
    
    out = _apply_sliced_conv(module.conv2, out, final_out_channels)
    out = _apply_sliced_bn(module.bn2, out)
    
    if hasattr(module, 'skip'):
        shortcut = apply_sliced_operation(module.skip, x, x.size(1), final_out_channels)
    else:
        shortcut = x if x.size(1) == final_out_channels else x[:, :final_out_channels]

    if out.shape == shortcut.shape:
        out += shortcut
        
    return module.act(out)


def _apply_sliced_conv(layer, x, target_out_channels):
    in_ch = x.size(1)
    
    if layer.groups > 1 and layer.groups == layer.in_channels:
        groups = in_ch
        out_ch = in_ch 
    else:
        groups = 1
        out_ch = target_out_channels

    # CRITICAL: Ensure we don't slice more channels than exist in supernet weights
    # This handles cases where supernet might be smaller than input if misconfigured,
    # though with the Init fix this shouldn't happen.
    max_in = layer.weight.size(1) * layer.groups
    if in_ch > max_in:
         raise RuntimeError(f"Input has {in_ch} channels but supernet layer {layer} only has {max_in} initialized. check Supernet Init.")

    sliced_weight = layer.weight[:out_ch, :in_ch // groups, :, :]
    sliced_bias = layer.bias[:out_ch] if layer.bias is not None else None

    return F.conv2d(
        x, sliced_weight, sliced_bias,
        layer.stride, layer.padding, layer.dilation, groups=groups
    )


def _apply_sliced_bn(layer, x):
    current_ch = x.size(1)
    return F.batch_norm(
        x,
        layer.running_mean[:current_ch],
        layer.running_var[:current_ch],
        layer.weight[:current_ch],
        layer.bias[:current_ch],
        layer.training,
        layer.momentum,
        layer.eps
    )


def _apply_sliced_se(module, x):
    in_ch = x.size(1)
    reduce_conv = module.se[1]
    super_in = reduce_conv.in_channels
    super_squeeze = reduce_conv.out_channels
    ratio = super_in / super_squeeze
    
    target_squeeze = max(1, int(in_ch / ratio))
    
    y = module.se[0](x)
    y = _apply_sliced_conv(reduce_conv, y, target_squeeze)
    y = module.se[2](y)
    expand_conv = module.se[3]
    y = _apply_sliced_conv(expand_conv, y, in_ch)
    y = module.se[4](y)
    
    return x * y


def _apply_sliced_mbconv(module, x, final_out_channels):
    layers = list(module.block)
    current_x = x
    in_ch = x.size(1)
    idx = 0
    
    # 1. Expansion
    if isinstance(layers[0], nn.Conv2d) and layers[0].kernel_size == (1, 1):
        if layers[0].out_channels > layers[0].in_channels:
            orig_in = layers[0].in_channels
            orig_exp = layers[0].out_channels
            ratio = orig_exp / orig_in
            expanded_ch = int(in_ch * ratio)
            
            current_x = _apply_sliced_conv(layers[0], current_x, expanded_ch)
            current_x = _apply_sliced_bn(layers[1], current_x)
            current_x = layers[2](current_x)
            idx = 3
    
    # 2. Depthwise
    dw_conv = layers[idx]
    current_x = _apply_sliced_conv(dw_conv, current_x, current_x.size(1))
    current_x = _apply_sliced_bn(layers[idx+1], current_x)
    current_x = layers[idx+2](current_x)
    idx += 3
    
    # 3. SE or Projection
    if idx < len(layers) and ("Squeeze" in layers[idx].__class__.__name__ or isinstance(layers[idx], nn.Sequential)):
        # Heuristic: if it's a generic sequential inside MBConv, it's likely SE
        if hasattr(layers[idx], 'se') or "Squeeze" in layers[idx].__class__.__name__:
             current_x = _apply_sliced_se(layers[idx], current_x)
        idx += 1

    # 4. Projection
    proj_conv = layers[idx]
    current_x = _apply_sliced_conv(proj_conv, current_x, final_out_channels)
    current_x = _apply_sliced_bn(layers[idx+1], current_x)
    
    if module.use_residual and in_ch == final_out_channels and x.shape == current_x.shape:
        return x + current_x
        
    return current_x


def _apply_sliced_sequential(module, x, final_out_channels):
    """
    Handles Sequential blocks (like DoubleConv).
    FIX: Propagates 'final_out_channels' to nested modules to ensure correct channel expansion.
    """
    layers = list(module) if isinstance(module, (nn.Sequential, nn.ModuleList)) else module
    conv_indices = [i for i, l in enumerate(layers) if isinstance(l, nn.Conv2d)]
    last_conv_idx = conv_indices[-1] if conv_indices else -1
    
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Conv2d):
            is_depthwise = (layer.groups > 1 and layer.groups == layer.in_channels)
            
            if is_depthwise:
                target = x.size(1)
            elif i == last_conv_idx:
                target = final_out_channels
            else:
                target = final_out_channels 
            
            x = _apply_sliced_conv(layer, x, target)
        elif isinstance(layer, nn.BatchNorm2d):
            x = _apply_sliced_bn(layer, x)
        else:
            # FIX: Do not pass x.size(1) as output. Pass final_out_channels.
            # This ensures that if 'layer' is a Sequential (like ConvNormAct), 
            # it knows we want to target the final width, effectively allowing expansion.
            x = apply_sliced_operation(layer, x, x.size(1), final_out_channels)
            
    return x