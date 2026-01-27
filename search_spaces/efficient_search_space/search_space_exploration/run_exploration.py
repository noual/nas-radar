import csv
import os
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")


import numpy as np
from omegaconf import OmegaConf

from search_spaces.efficient_search_space.radar_node import FrugalRadarNode
from search_spaces.radar import Radar



import random
import time
from tqdm import trange, tqdm
from training_free_metrics.ntk.compute_ntk import compute_ntk
from training_free_metrics.zen_nas.zen_nas import ZenNAS
from training_free_metrics.naswot.naswot import NASWOT
from data.radar_dataset import RadarDavaDataset

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda:0', lr=1e-3, verbose=False):
    """Train the model using Dice Loss"""
    model = model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False) if verbose else train_loader
        
        for batch_idx, (data, target) in enumerate(iterator):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if verbose and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses


if __name__ == "__main__":
    # Import dataset
    sys.path.append("../../data")
    
    # Configuration
    NUM_TRIALS = 10000  # Number of architectures to evaluate
    NUM_EPOCHS = 20  # Training epochs per architecture
    CSV_PATH = "exploration_with_training.csv"
    
    # Create dataset and data loaders (load once, reuse for all trials)
    print("Loading RadarDavaDataset...")
    dataset = RadarDavaDataset(root_dir="../data/train_bth/mat", max_idx=1000, batch_size=8)
    train_loader, val_loader, test_loader = dataset.generate_loaders(test_split=0.8, val_split=0.8)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Create sample data for NTK/Zen-NAS/NASWOT
    print("Preparing sample data for metrics...")
    x_sample = torch.randn(10, 1, 16, 16).to("cuda:0")
    x_cpu = torch.randn(10, 1, 16, 16).to("cpu")
    config = OmegaConf.create({
    "problem": {
        "n_objectives": 2,
        "dataset_path": "../data/train_bth/mat",
        "batch_size": 8,
        "supernet": {
            "in_channels": 1,
            "initial_channels": 8,
            "channel_options": [8, 16, 32, 64, 128, 256],
            "num_encoder_stages": 4,
            "num_nodes": 4,
            "n_steps": 5
            },

    },
    "device": "cuda:0",
    "seed": 42
    })
    
    # Main exploration loop
    for i in trange(NUM_TRIALS, desc="Architecture Search"):
        try:
            # Create random architecture
            problem = Radar(config)
            node = FrugalRadarNode(problem)
            
            while not node.is_terminal:
                actions = node.get_actions_tuples()
                action = random.choice(actions)
                name, value = action
                node.play_action(name, value)
                    
            model = problem.supernet.sample(node)
            model.to(config.device)
            print(np.unique([param.device.type for param in model.parameters()]))
            output = model(x_sample)
            model.to("cuda:0")
            
            architecture_str = node.to_str()
            print(f"\n[{i+1}/{NUM_TRIALS}] Evaluating architecture: {architecture_str}")
            num_params = sum(p.numel() for p in model.parameters())
            
            # Compute inference time on GPU
            with torch.no_grad():
                # Warmup
                _ = model(x_sample)
                torch.cuda.synchronize()
                
                # Timing
                start_time = time.time()
                for _ in range(100):
                    _ = model(x_sample)
                torch.cuda.synchronize()
                end_time = time.time()
                
                inference_time_gpu = (end_time - start_time) / 100

            # Compute inference time on CPU
            model.to("cpu")
            with torch.no_grad():
                # Warmup
                _ = model(x_cpu)
                
                # Timing
                start_time = time.time()
                for _ in range(100):
                    _ = model(x_cpu)
                end_time = time.time()
                
                inference_time_cpu = (end_time - start_time) / 100

            model.to("cuda:0")
            
            # Compute NTK metrics
            print(f"[{i+1}/{NUM_TRIALS}] Computing NTK...")
            ntk_model = deepcopy(model)
            ntk_matrix = compute_ntk(ntk_model, x_sample, chunk_size=1, use_fp16=False)
            lambda_0 = torch.linalg.eigvalsh(ntk_matrix).min().item()
            condition_number = torch.linalg.cond(ntk_matrix).item()
            
            # Compute Zen-NAS score
            print(f"[{i+1}/{NUM_TRIALS}] Computing Zen-NAS...")
            nas_evaluator = ZenNAS(data_loader=[], in_channels=1, resolution=128)
            zen_nas_score = nas_evaluator.compute_nas_score(model, repeat=3, fp16=False)["avg_nas_score"]
            
            # Compute NASWOT score
            print(f"[{i+1}/{NUM_TRIALS}] Computing NASWOT...")
            naswot_evaluator = NASWOT(data_loader=torch.utils.data.TensorDataset(x_sample, torch.zeros(x_sample.size(0))))
            naswot_score = naswot_evaluator.score(model)
            
            # Train the model
            print(f"[{i+1}/{NUM_TRIALS}] Training for {NUM_EPOCHS} epochs...")
            train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                                   num_epochs=NUM_EPOCHS, device='cuda:0', 
                                                   lr=5e-4, verbose=False)
            
            final_train_loss = min(train_losses)
            final_val_loss = min(val_losses)
            
            # Prepare row for CSV
            row = {
                "architecture": architecture_str,
                "params": num_params,
                "inference_time_gpu": inference_time_gpu,
                "inference_time_cpu": inference_time_cpu,
                "ntk_lambda_0": lambda_0,
                "ntk_condition_number": condition_number,
                "zen_nas_score": zen_nas_score,
                "naswot_score": naswot_score,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss
            }
            
            # Write to CSV
            file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
            with open(CSV_PATH, "a", newline="") as f:
                fieldnames = ["architecture", "params", "inference_time_gpu", "inference_time_cpu", "ntk_lambda_0", 
                            "ntk_condition_number", "zen_nas_score", "naswot_score", 
                            "final_train_loss", "final_val_loss"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            
            print(f"[{i+1}/{NUM_TRIALS}] Completed - Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f}")
            
            # Clean up GPU memory
            del model, node, ntk_matrix
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[{i+1}/{NUM_TRIALS}] Error: {e}")
            continue
    
    print(f"\nExploration complete! Results saved to {CSV_PATH}")
