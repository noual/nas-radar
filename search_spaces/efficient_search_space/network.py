import copy
import itertools
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(".")
sys.path.append("..")

from efficient_search_space.operations import OPERATIONS

def apply_sliced_operation(module, x, in_channels, out_channels):
    """Apply operation with dynamically sliced weights - no copying, preserves gradients."""
    # Extract layers
    conv_layer = None
    bn_layer = None
    relu_layer = None
    
    for layer in module:
        if isinstance(layer, nn.Identity):
            if in_channels == out_channels:
                return x
            elif in_channels < out_channels:
                return F.pad(x, (0,0,0,0,0,out_channels - in_channels)) 
            else:
                return x[:, :out_channels, :, :]
        if isinstance(layer, nn.Conv2d):
            conv_layer = layer
        elif isinstance(layer, nn.BatchNorm2d):
            bn_layer = layer
        elif isinstance(layer, nn.ReLU):
            relu_layer = layer
    
    # Apply convolution with sliced weights
    sliced_weight = conv_layer.weight[:out_channels, :in_channels, :, :]
    print(f"Sliced weight shape: {sliced_weight.shape}")
    sliced_bias = conv_layer.bias[:out_channels] if conv_layer.bias is not None else None
    
    x = F.conv2d(x, sliced_weight, sliced_bias, 
                 conv_layer.stride, conv_layer.padding, 
                 conv_layer.dilation, conv_layer.groups)
    
    # Apply batch normalization with sliced parameters
    if bn_layer is not None:
        x = F.batch_norm(
            x,
            bn_layer.running_mean[:out_channels],
            bn_layer.running_var[:out_channels], 
            bn_layer.weight[:out_channels],
            bn_layer.bias[:out_channels],
            bn_layer.training,
            bn_layer.momentum,
            bn_layer.eps
        )
    
    # Apply ReLU
    if relu_layer is not None:
        x = F.relu(x, inplace=True)
    
    return x

class MaximalFrugalRadarNetwork(nn.Module):
    
    def __init__(self, in_channels=3, initial_channels=8, max_channels=512, num_encoder_stages=3, device="cpu"):
        super().__init__()
        self.out_channels = None
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

        for i in range(num_encoder_stages):
            out_channels = max_channels
            module_dict = nn.ModuleDict()
            for op_name, operation in OPERATIONS.items():
                module_dict[op_name] = operation(in_channels, out_channels)
            self.encoder.append(module_dict)
            in_channels = out_channels

        # Bottleneck
        for op_name, operation in OPERATIONS.items():
            self.bottleneck[op_name] = operation(in_channels, out_channels)
        in_channels = out_channels
        
        # Decoder
        for i in range(num_encoder_stages):
            out_channels = max_channels
            module_dict = nn.ModuleDict()
            for op_name, operation in OPERATIONS.items():
                module_dict[op_name] = operation(in_channels, out_channels)
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
            x = apply_sliced_operation(module_dict[op_name], x, in_channels=x.size(1), out_channels=channels)

        sliced_weight = self.final_conv.weight[:self.out_channels, :x.size(1), :, :]
        sliced_bias = self.final_conv.bias[:self.out_channels] if self.final_conv.bias is not None else None
        
        x = F.conv2d(x, sliced_weight, sliced_bias, 
                    self.final_conv.stride, self.final_conv.padding, 
                    self.final_conv.dilation, self.final_conv.groups)
        return x

    

class FrugalRadarNetwork(nn.Module):

    def __init__(self, in_channels=3, initial_channels=8, out_channels=1, num_encoder_stages=2, device="cpu"):
        super().__init__()
        self.out_channels = out_channels
        self.channel_list = {"encoder": [None for e in range(num_encoder_stages)], "bottleneck": None, "decoder": [None for e in range(num_encoder_stages)]}
        self.encoder = nn.ModuleList()
        self.bottleneck = None
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Initial convolution to map input to initial_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        in_channels = initial_channels
        self.final_conv = None
        self.device = device

    def play_action(self, action):
        if not action[0].endswith("channels"):
            if action[0].startswith("encoder"):
                stage_idx = int(action[0].split("_")[1])
                # Assert that the channel has already been set
                if self.channel_list["encoder"][stage_idx] is None:
                    raise ValueError(f"Encoder stage {stage_idx} channels must be set before setting operation.")
                op_name = action[1]
                in_ch = self.stem[0].out_channels if stage_idx == 0 else self.channel_list["encoder"][stage_idx - 1]
                out_ch = self.channel_list["encoder"][stage_idx]
                operation = OPERATIONS[op_name](in_ch, out_ch)
                self.encoder.append(operation)
            elif action[0].startswith("bottleneck"):
                op_name = action[1]
                # Assert that the channel has already been set
                if self.channel_list["bottleneck"] is None:
                    raise ValueError(f"Bottleneck channels must be set before setting operation.")
                in_ch = self.channel_list["encoder"][-1]
                out_ch = self.channel_list["bottleneck"]
                operation = OPERATIONS[op_name](in_ch, out_ch)
                self.bottleneck = operation
            elif action[0].startswith("decoder"):  
                stage_idx = int(action[0].split("_")[1])
                # Assert that the channel has already been set
                if self.channel_list["decoder"][stage_idx] is None:
                    raise ValueError(f"Decoder stage {stage_idx} channels must be set before setting operation.")
                
                op_name = action[1]
                in_ch = self.channel_list["bottleneck"] if stage_idx == 0 else self.channel_list["decoder"][stage_idx - 1]
                in_ch += self.channel_list["encoder"][-(stage_idx + 1)]  # due to skip connection
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
        # Only final conv remains
        if self.final_conv is None:
            self.final_conv = nn.Conv2d(self.channel_list["decoder"][-1], 1, kernel_size=1)
            self.to(device=self.device)
        return True

    def forward(self, x):
        assert self.is_terminal, "Network is not fully defined."
        x = self.stem(x)
        skip_connections = []
        # Encoder
        for i, module in enumerate(self.encoder):
            x = module(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for i, module in enumerate(self.decoder):
            x = self.up(x)
            skip_conn = skip_connections[i]
            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:])
            x = torch.cat((skip_conn, x), dim=1)
            x = module(x)


        sliced_weight = self.final_conv.weight[:self.out_channels, :x.size(1), :, :]
        sliced_bias = self.final_conv.bias[:self.out_channels] if self.final_conv.bias is not None else None
        
        x = F.conv2d(x, sliced_weight, sliced_bias, 
                    self.final_conv.stride, self.final_conv.padding, 
                    self.final_conv.dilation, self.final_conv.groups)
        return x


if __name__ == "__main__":
    # Example usage
    model = MaximalFrugalRadarNetwork(in_channels=3, initial_channels=8, max_channels=512, num_encoder_stages=2, device="cpu")
    x = torch.randn(1, 3, 128, 128)
    selected_ops = {
        'encoder': ['conv5x5', 'conv3x3'],
        'bottleneck': 'conv5x5',
        'decoder': ['conv3x3', 'conv3x3']
    }
    channel_config = {
        'encoder': [128, 128],
        'bottleneck': 512,
        'decoder': [128, 32]
    }
    output = model(x, selected_ops, channel_config)
    print(output.shape)  # Expected output shape: (1, 1, 128, 128)

    print(f"=========================================")

    # Test FrugalRadarNetwork
    frugal_model = FrugalRadarNetwork(in_channels=3, initial_channels=8, out_channels=1, num_encoder_stages=2, device="cpu")
    actions = [
        {0: "encoder_0_channels", 1: 64},
        {0: "encoder_0_operation", 1: "conv3x3"},
        {0: "encoder_1_channels", 1: 128},
        {0: "encoder_1_operation", 1: "conv3x3"},
        {0: "bottleneck_channels", 1: 256},
        {0: "bottleneck_operation", 1: "conv3x3"},
        {0: "decoder_0_channels", 1: 64},
        {0: "decoder_0_operation", 1: "conv3x3"},
        {0: "decoder_1_channels", 1: 32},
        {0: "decoder_1_operation", 1: "conv3x3"},
    ]
    for action in actions:
        frugal_model.play_action(action)
    frugal_output = frugal_model(x)
    print(frugal_output.shape)  # Expected output shape: (1, 1, 128, 128)