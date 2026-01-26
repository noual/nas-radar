import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class ZenNAS:
    def __init__(self, data_loader, in_channels=1, resolution=128):
        self.batch_size=8
        self.data_loader = DataLoader(data_loader, batch_size=self.batch_size, shuffle=False)
        self.mixup_gamma = 1e-2
        self.gpu = 0
        self.in_channels = in_channels
        self.resolution = resolution

    def network_weight_gaussian_init(self, net: nn.Module):
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    continue

        return net

    def compute_nas_score(self, model, repeat=1, fp16=False):
        info = {}
        nas_score_list = []
        if self.gpu is not None:
            device = torch.device('cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')

        if fp16:
            dtype = torch.half
        else:
            dtype = torch.float32

        with torch.no_grad():
            for repeat_count in range(repeat):
                self.network_weight_gaussian_init(model)
                input = torch.randn(size=[self.batch_size, self.in_channels, self.resolution, self.resolution], device=device, dtype=dtype)
                input2 = torch.randn(size=[self.batch_size, self.in_channels, self.resolution, self.resolution], device=device, dtype=dtype)
                mixup_input = input + self.mixup_gamma * input2
                output = model.forward(input)
                mixup_output = model.forward(mixup_input)

                nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
                nas_score = torch.mean(nas_score)

                # compute BN scaling
                log_bn_scaling_factor = 0.0
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling_factor += torch.log(bn_scaling_factor)
                    pass
                pass
                nas_score = torch.log(nas_score) + log_bn_scaling_factor
                nas_score_list.append(float(nas_score))


        std_nas_score = np.std(nas_score_list)
        avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
        avg_nas_score = np.mean(nas_score_list)


        info['avg_nas_score'] = float(avg_nas_score)
        info['std_nas_score'] = float(std_nas_score)
        info['avg_precision'] = float(avg_precision)
        return info

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append("../..")
    from search_spaces.radar_node import RadarNode
    import random


    node = RadarNode(in_channels=3, initial_channels=8, channel_options=[8, 16, 32, 64], num_encoder_stages=3, num_nodes=4)
    while not node.is_terminal:
        actions = node.get_actions_tuples()
        action = random.choice(actions)  # Just pick the first action for testing
        node.play_action(action)
    
    model = node.network
    model.prune_cells()
    model.to("cuda:0")
    x = torch.randn(10, 3, 32, 32).to("cuda:0")  # Example input batch of size 10
    nas_evaluator = ZenNAS(data_loader=[], in_channels=3, resolution=32)
    nas_info = nas_evaluator.compute_nas_score(model, repeat=3, fp16=False)
    print("NAS Score Info:", nas_info)  