import sys
sys.path.append("..")

from data.radar_dataset import RadarDavaDataset
from search_space.supernet import SuperNet
from helpers.utils import DiceLoss
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
        self.supernet = SuperNet(
            channel_options=config.problem.supernet.channel_options,
            in_channels=config.problem.supernet.in_channels,
            initial_channels=config.problem.supernet.initial_channels,
            num_encoder_stages=config.problem.supernet.num_encoder_stages,
            num_nodes=config.problem.supernet.num_nodes, 
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
            selected_ops = self.supernet.get_selected_ops_from_node(node)
            outputs = self.supernet.maximal_net(inputs, selected_ops=selected_ops)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            del outputs, loss
        avg_loss = total_loss / self.n_steps

        # Latency: sample the supernet, prune it and measure latency
        with torch.no_grad():
            sample = self.supernet.sample(node).to(self.device)
            sample.eval()
            sample.prune_cells()
            # Warmup
            _ = sample(inputs)
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(10):
                _ = sample(inputs)
            t1 = time.time()
            torch.cuda.synchronize()
            latency = (t1 - t0) / 10.0
        del sample, inputs, targets
        torch.cuda.empty_cache()
        return [avg_loss, latency]