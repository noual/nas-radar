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
        self._copy_weights_to_subnet(subnet)

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
        # Copy weights from maximal net to subnet
        for (name_max, module_max), (name_sub, module_sub) in zip(self.maximal_net.named_modules(), subnet.named_modules()):
            if name_max == name_sub:
                if isinstance(module_max, nn.Conv2d) and isinstance(module_sub, nn.Conv2d):
                    # Copy weights for Conv2d layers
                    out_channels = module_sub.out_channels
                    in_channels = module_sub.in_channels
                    module_sub.weight.data = module_max.weight.data[:out_channels, :in_channels, :, :].clone()
                    if module_max.bias is not None:
                        module_sub.bias.data = module_max.bias.data[:out_channels].clone()
                elif isinstance(module_max, nn.BatchNorm2d) and isinstance(module_sub, nn.BatchNorm2d):
                    # Copy weights for BatchNorm2d layers
                    num_features = module_sub.num_features
                    module_sub.weight.data = module_max.weight.data[:num_features].clone()
                    module_sub.bias.data = module_max.bias.data[:num_features].clone()
                    module_sub.running_mean = module_max.running_mean[:num_features].clone()
                    module_sub.running_var = module_max.running_var[:num_features].clone()