import csv
import sys, os
import torch
import torch.optim as optim
import time
from tqdm import trange, tqdm
from copy import deepcopy
sys.path.append("..")
from search_space.radar_node import RadarNode
import random



if __name__ == "__main__":
    # Import dataset
    sys.path.append("../data")
    from radar_dataset import RadarDavaDataset
    
    # Configuration
    NUM_TRIALS = 1000000  # Number of architectures to evaluate
    NUM_EPOCHS = 20  # Training epochs per architecture
    CSV_PATH = "random_architectures.csv"

    x_sample_gpu = torch.randn(10, 1, 128, 128).to("cuda:0")
    x_sample = x_sample_gpu.cpu()
    
    # Main exploration loop
    for i in trange(NUM_TRIALS, desc="Architecture Search"):
        try:
            # Create random architecture
            node = RadarNode(in_channels=1, initial_channels=8, 
                           channel_options=[8, 16, 32, 64], 
                           num_encoder_stages=3, num_nodes=4)
            
            while not node.is_terminal:
                actions = node.get_actions_tuples()
                action = random.choice(actions)
                node.play_action(action)
            
            model = node.network
            model.prune_cells()
            model_gpu = deepcopy(model).to("cuda:0")
            model.to("cpu")
            
            architecture_str = node.to_str()
            num_params = sum(p.numel() for p in model.parameters())
            
            # Compute inference time
            with torch.no_grad():
                # Warmup
                _ = model(x_sample)
                torch.cuda.synchronize()
                
                # Timing
                start_time = time.time()
                for _ in range(10):
                    _ = model(x_sample)
                torch.cuda.synchronize()
                end_time = time.time()
                
                inference_time = (end_time - start_time) / 10

                # Warmup
                _ = model_gpu(x_sample_gpu)
                torch.cuda.synchronize()
                
                # Timing
                start_time = time.time()
                for _ in range(10):
                    _ = model_gpu(x_sample_gpu)
                torch.cuda.synchronize()
                end_time = time.time()
                
                inference_time_gpu = (end_time - start_time) / 10
                    # Prepare row for CSV
            row = {
                "architecture": architecture_str,
                "params": num_params,
                "inference_time_cpu": inference_time,
                "inference_time_gpu": inference_time_gpu,
            }
            
            # Write to CSV
            file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
            with open(CSV_PATH, "a", newline="") as f:
                fieldnames = ["architecture", "params", "inference_time_cpu", "inference_time_gpu"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            
            # Clean up GPU memory
            del model, node, model_gpu
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[{i+1}/{NUM_TRIALS}] Error: {e}")
            continue