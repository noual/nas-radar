from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(".")
sys.path.append("..")

from search_spaces.efficient_search_space.network import FrugalRadarNetwork, MaximalFrugalRadarNetwork

class FrugalSuperNet(nn.Module):

    def __init__(self, 
                 in_channels,
                 initial_channels,
                 channel_options,
                 num_encoder_stages=3,
                 device="cpu",
                 ):
        super().__init__()

        self.channel_options = sorted(channel_options)
        self.max_channels = 2*max(channel_options)
        self.in_channels = in_channels
        self.initial_channels = initial_channels
        self.num_encoder_stages = num_encoder_stages
        self.device = device

        # Create the maximal network
        self.maximal_net = MaximalFrugalRadarNetwork(in_channels=in_channels,
                                                 initial_channels=initial_channels,
                                                 max_channels=self.max_channels, 
                                                 num_encoder_stages=num_encoder_stages,
                                                 device=device)
        self.maximal_net.to(device)
        
    def forward(self, x, selected_ops, channel_config):
        return self.maximal_net(x, selected_ops, channel_config)
    
    def sample(self, node):
        if not node.is_terminal:
            raise ValueError("Can only sample from terminal nodes.")
        
        # Get channel and operations configuration
        list_actions = node.state  # get_state returns the path
        subnet = self._create_subnet(list_actions)
        self._copy_weights_to_subnet(subnet, node.get_selected_ops())

        return subnet
    
    def _create_subnet(self, list_actions):

        subnet = FrugalRadarNetwork(
            in_channels=self.maximal_net.in_channels,
            initial_channels=self.maximal_net.initial_channels,
            num_encoder_stages=self.num_encoder_stages,
            device=self.device
        )

        for action in list_actions:
            subnet.play_action(action)
            
        return subnet

    def _copy_weights_to_subnet(self, subnet: FrugalRadarNetwork):
        # 1. Copy Independent Modules (Stem, Final Conv)
        # These names match, so we can use the simple dictionary lookup
        max_dict = dict(self.maximal_net.named_modules())
        
        for name_sub, module_sub in subnet.named_modules():
            # Handle Stem and Final Conv (names match directly)
            if name_sub in max_dict:
                self._copy_module_weights(max_dict[name_sub], module_sub)
                continue
            
            # 2. Handle Encoder/Decoder/Bottleneck Mismatches
            # Subnet Name: encoder.0.0.weight
            # Target Supernet Name: encoder.0.{OP_NAME}.0.weight
            
            parts = name_sub.split('.')
            if len(parts) > 2 and parts[0] in ['encoder', 'decoder', 'bottleneck']:
                stage_type = parts[0]
                
                # Retrieve the operation name from the Subnet's history/state
                # We need to look up which op was chosen for this stage.
                # Since 'subnet' doesn't easily expose the chosen op name per module,
                # we rely on the fact that the Subnet was just built from the 'node'.
                # However, a cleaner way within this function is to assume specific structure.
                
                # Let's rely on the fact that Subnet was built sequentially.
                # We can't easily guess 'conv_3x3' from just 'encoder.0'.
                pass 

        # BETTER APPROACH: Iterate over the stages directly
        
        # A. Copy Stem
        self._recursive_copy(self.maximal_net.stem, subnet.stem)

        # B. Copy Encoder
        # subnet.encoder is a ModuleList of selected ops
        # maximal_net.encoder is a ModuleList of ModuleDicts
        for i, sub_op in enumerate(subnet.encoder):
            # We need to find which op this is. 
            # We can check the Subnet's channel_list or just check which key in MaxNet matches the structure.
            # But the most robust way is to pass the 'node' or 'selected_ops' to this function.
            # Assuming we don't change the signature, we can infer it:
            
            # Find the op name by checking which supernet op has the same structure/type? No, too risky.
            # WE MUST USE THE NODE/ACTIONS. 
            # However, looking at 'sample(node)', you create the subnet. 
            # You should pass the 'list_actions' or 'node' to _copy_weights_to_subnet.
            pass

    # --- SIMPLIFIED IMPLEMENTATION ---
    # Update sample() to pass the node/selected_ops, OR use the logic below 
    # that iterates the node state if you can access it. 
    # Assuming we modify sample() to: self._copy_weights_to_subnet(subnet, node)
    
    def sample(self, node):
        if not node.is_terminal:
            raise ValueError("Can only sample from terminal nodes.")
        
        list_actions = node.state 
        subnet = self._create_subnet(list_actions)
        
        # FIX: Pass the selected ops to the copy function
        selected_ops = node.get_selected_ops() 
        self._copy_weights_to_subnet(subnet, selected_ops) # <--- Update call here

        return subnet

    def _copy_weights_to_subnet(self, subnet: FrugalRadarNetwork, selected_ops: dict):
        
        # 1. Stem (Direct Copy)
        self._recursive_copy(self.maximal_net.stem, subnet.stem)
        
        # 2. Encoder
        for i, sub_module in enumerate(subnet.encoder):
            op_name = selected_ops['encoder'][i]
            max_module = self.maximal_net.encoder[i][op_name]
            self._recursive_copy(max_module, sub_module)

        # 3. Bottleneck
        op_name = selected_ops['bottleneck']
        self._recursive_copy(self.maximal_net.bottleneck[op_name], subnet.bottleneck)

        # 4. Decoder
        for i, sub_module in enumerate(subnet.decoder):
            op_name = selected_ops['decoder'][i]
            max_module = self.maximal_net.decoder[i][op_name]
            self._recursive_copy(max_module, sub_module)
            
        # 5. Final Conv (Direct Copy)
        # Note: FrugalRadarNetwork initializes final_conv only on first forward or manually.
        # Ensure it is initialized before copying.
        if subnet.final_conv is None:
             # Force init if needed, or assume it's done. 
             # Based on your code, it's done in is_terminal property check or forward.
             # You might need to manually init it here if it's None.
             pass 
        if subnet.final_conv is not None:
             self._recursive_copy(self.maximal_net.final_conv, subnet.final_conv)

    def _recursive_copy(self, source_module, target_module):
        # Iterate over all sub-modules (convs, bns) and copy weights
        source_dict = dict(source_module.named_modules())
        target_dict = dict(target_module.named_modules())
        
        for name, sub_mod in target_dict.items():
            if name in source_dict:
                src_mod = source_dict[name]
                if isinstance(sub_mod, nn.Conv2d) and isinstance(src_mod, nn.Conv2d):
                    # Slicing Logic
                    out_ch = sub_mod.out_channels
                    in_ch = sub_mod.in_channels
                    groups = sub_mod.groups
                    
                    # Sliced Copy
                    sub_mod.weight.data = src_mod.weight.data[:out_ch, :in_ch // groups, :, :].clone()
                    if src_mod.bias is not None and sub_mod.bias is not None:
                        sub_mod.bias.data = src_mod.bias.data[:out_ch].clone()
                        
                elif isinstance(sub_mod, nn.BatchNorm2d) and isinstance(src_mod, nn.BatchNorm2d):
                    num = sub_mod.num_features
                    sub_mod.weight.data = src_mod.weight.data[:num].clone()
                    sub_mod.bias.data = src_mod.bias.data[:num].clone()
                    sub_mod.running_mean = src_mod.running_mean[:num].clone()
                    sub_mod.running_var = src_mod.running_var[:num].clone()