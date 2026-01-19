import csv
import os
import sys
sys.path.append(".")
sys.path.append("..")

from radar_node import RadarNode
import random
import time
from tqdm import trange
from training_free_metrics.ntk.compute_ntk import compute_ntk
from training_free_metrics.zen_nas.zen_nas import ZenNAS
from training_free_metrics.naswot.naswot import NASWOT
import torch


if __name__ == "__main__":


    csv_path = "exploration.csv"


    for i in trange(1000):
        node = RadarNode(in_channels=3, initial_channels=8, channel_options=[8, 16, 32, 64], num_encoder_stages=3, num_nodes=5)
        while not node.is_terminal:
            actions = node.get_actions_tuples()
            action = random.choice(actions)
            node.play_action(action)
        model = node.network
        model.prune_cells()
        model.to("cuda:0")
        with torch.no_grad():
            x = torch.randn(10, 3, 16, 16).to("cuda:0")
            #Warmup
            y = model(x)
            #Timing
            start_time = time.time()
            for j in range(10):
                y = model(x)
            end_time = time.time()

            times = (end_time - start_time) / 10

        # NTK
        ntk_matrix = compute_ntk(node.network, x, chunk_size=1, use_fp16=True)
        lambda_0 = torch.linalg.eigvalsh(ntk_matrix).min().item()
        condition_number = torch.linalg.cond(ntk_matrix).item()

        #Zen-NAS
        nas_evaluator = ZenNAS(data_loader=[], in_channels=3, resolution=32)
        nas_info = nas_evaluator.compute_nas_score(node.network, repeat=3, fp16=False)["avg_nas_score"]

        # NASWOT
        naswot_evaluator = NASWOT(data_loader=torch.utils.data.TensorDataset(x, torch.zeros(x.size(0))))
        naswot_score = naswot_evaluator.score(node.network)
        row = {
            "architecture": node.to_str(),
            "params": sum(p.numel() for p in node.network.parameters()),
            "inference_time": times,
            "ntk_lambda_0": lambda_0,
            "ntk_condition_number": condition_number,
            "zen_nas_score": nas_info,
            "naswot_score": naswot_score
        }
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["architecture", "params", "inference_time", "ntk_lambda_0", "ntk_condition_number", "zen_nas_score", "naswot_score"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

