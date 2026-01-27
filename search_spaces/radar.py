import sys

from omegaconf import OmegaConf

sys.path.append("..")
sys.path.append(".")
sys.path.append("../..")
from search_spaces.efficient_search_space.supernet import FrugalSuperNet

from data.radar_dataset import RadarDavaDataset
from search_spaces.dag_search_space.supernet import SuperNet
from utils.helpers import DiceLoss
import time
import torch
from types import SimpleNamespace
import numpy as np

class Radar:

    def __init__(self, config):
        self.n_objectives = config.problem.n_objectives
        self.seed = config.seed
        self.nadir = [1, 0.5][:self.n_objectives] # 1 = max loss, 0.5 seconds = probably the max latency on a GPU 
        self._move_coder = lambda arch_str, move: tuple(arch_str) + (move,)
        self.supernet = FrugalSuperNet(
            channel_options=config.problem.supernet.channel_options,
            in_channels=config.problem.supernet.in_channels,
            initial_channels=config.problem.supernet.initial_channels,
            num_encoder_stages=config.problem.supernet.num_encoder_stages,
            device=config.device
        )
        self.supernet.maximal_net.to(config.device)
        self.dataset_path = config.problem.dataset_path
        self.dataset = RadarDavaDataset(root_dir=self.dataset_path, batch_size=config.problem.batch_size, has_distance=True)
        self.train_loader, self.val_loader, self.test_loader = self.dataset.generate_loaders(test_split=0.8, val_split=0.8)
        self.n_steps = config.problem.supernet.n_steps
        self.criterion = DiceLoss()
        self.device = config.device
        self.optimizer = torch.optim.Adam(self.supernet.maximal_net.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4)

    def _evaluate(self, node) -> list:
        # Accuracy: run the supernet on k batches 
        self.supernet.train()
        total_loss = 0.0
        for k in range(self.n_steps):
            batch = next(iter(self.train_loader))
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            selected_ops = node.get_selected_ops()
            channel_config = node.get_channel_config()
            outputs = self.supernet.maximal_net(inputs, selected_ops=selected_ops, channel_config=channel_config)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            del outputs, loss
        avg_loss = total_loss / self.n_steps

        # Latency: sample the supernet, prune it and measure latency
        x_cpu = torch.randn(1, self.supernet.in_channels, 128, 128).to("cpu")
        x_gpu = x_cpu.to(self.device)
        with torch.no_grad():
            sample = self.supernet.sample(node)
            sample.eval()
            # Warmup
            _ = sample(x_gpu)
            sample.to("cpu")
            inputs = x_cpu.to("cpu")
            t0 = time.time()
            for i in range(10):
                _ = sample(inputs)
            t1 = time.time()
            latency = (t1 - t0) / 10.0
        del sample, inputs, targets
        return [avg_loss, latency]
    
if __name__ == "__main__":
    # Example usage
    import sys
    import random
    sys.path.append("..")
    sys.path.append(".")
    sys.path.append("../..")
    from efficient_search_space.supernet import FrugalSuperNet
    from efficient_search_space.network import MaximalFrugalRadarNetwork
    from efficient_search_space.radar_node import FrugalRadarNode

    config = OmegaConf.create({
    "problem": {
        "n_objectives": 2,
        "dataset_path": "../data/train_bth/mat",
        "batch_size": 8,
        "supernet": {
            "in_channels": 3,
            "initial_channels": 8,
            "channel_options": [8, 16, 32, 64, 128, 256],
            "num_encoder_stages": 3,
            "n_steps": 5
            },

    },
    "device": "cuda:0",
    "seed": 42
    })

    supernet = FrugalSuperNet(in_channels=config.problem.supernet.in_channels,
                              initial_channels=config.problem.supernet.initial_channels,
                              channel_options=config.problem.supernet.channel_options, num_encoder_stages=config.problem.supernet.num_encoder_stages, device=config.device)
    x = torch.randn(1, 3, 128, 128).to(config.device)


    problem = Radar(config)
    node = FrugalRadarNode(problem)
    while not node.is_terminal:
        actions = node.get_actions_tuples()
        action = random.choice(actions)
        print("Playing action:", action)
        name, value = action
        node.play_action(name, value)

    subnet = supernet.sample(node)
    subnet.to(config.device)
    print(np.unique([param.device.type for param in subnet.parameters()]))
    output = subnet(x)
    print("Output shape from sampled subnet:", output.shape)  # Expected output shape: (1, 1, 128, 128)

    # Get number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    supernet_params = count_parameters(problem.supernet.maximal_net)
    subnet_params = count_parameters(subnet)

    print(f"Supernet parameters: {supernet_params:,}")
    print(f"Sampled subnet parameters: {subnet_params:,}")
    print(f"Parameter reduction: {(1 - subnet_params/supernet_params)*100:.2f}%")
            
    from utils.helpers import simple_benchmark_model
    from utils.models.unet import UNet


    t = simple_benchmark_model(subnet, input_shape=(1, 3, 128, 128), device=config.device)
    print(f"Sampled subnet inference time over 100 iterations: {t:.4f}s ({t/100*1000:.2f}ms per prediction)")

    t_cpu = simple_benchmark_model(subnet, input_shape=(1, 3, 128, 128), device="cpu")
    print(f"Supernet inference time on CPU over 100 iterations: {t_cpu:.4f}s ({t_cpu/100*1000:.2f}ms per prediction)")

    unet = UNet(in_channels=3, initial_channels=8, features=[16, 32, 64, 128]).to(config.device)
    t_unet = simple_benchmark_model(unet, input_shape=(1, 3, 128, 128), device=config.device)
    print(f"UNet inference time on GPU over 100 iterations: {t_unet:.4f}s ({t_unet/100*1000:.2f}ms per prediction)")

    t_unet_cpu = simple_benchmark_model(unet, input_shape=(1, 3, 128, 128), device="cpu")
    print(f"UNet inference time on CPU over 100 iterations: {t_unet_cpu:.4f}s ({t_unet_cpu/100*1000:.2f}ms per prediction)")