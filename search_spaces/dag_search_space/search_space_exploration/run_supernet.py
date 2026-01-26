import sys
sys.path.append("..")
sys.path.append("../..")
from search_spaces.supernet import SuperNet
import torch
import torch.nn as nn
from helpers.utils import DiceLoss, configure_seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import time
configure_seaborn()


if __name__ == "__main__":
    channel_options = [8, 16, 32, 64]
    in_channels = 1
    initial_channels = 8
    num_encoder_stages = 3
    num_nodes = 5
    supernet = SuperNet(channel_options=channel_options,
                        in_channels=in_channels,
                        initial_channels=initial_channels,
                        num_encoder_stages=num_encoder_stages,
                        num_nodes=num_nodes, 
                        device="cuda:0")
    print(f"Number of parameters in SuperNet: {sum(p.numel() for p in supernet.parameters() if p.requires_grad)}")

    radar_dataset_path = "../../data/train_bth/mat"
    from data.radar_dataset import RadarDavaDataset
    import csv
    import os
    dataset = RadarDavaDataset(root_dir=radar_dataset_path, batch_size=8, has_distance=True)
    train_loader, val_loader, test_loader = dataset.generate_loaders(test_split=0.8, val_split=0.8)
    print(f"Dataset loaded with {len(dataset)} samples.")

    dict_metrics = {"train_loss": []}
    optimizer = torch.optim.Adam(supernet.parameters(), lr=1e-4)
    criterion = DiceLoss()
    
    # Move supernet to GPU for training
    supernet.to("cuda:0")

    for i in range(10000):
        # Sample a random architecture (configuration only, no weight copy)
        t1 = time.time()
        node = supernet.sample_random_architecture()
        t2 = time.time()
        print(f"Sampling Time taken: {t2 - t1:.4f}s")
        # Get selected operations from the sampled architecture
        selected_ops = supernet.get_selected_ops_from_node(node)
        t3 = time.time()
        print(f"Get Selected Ops Time taken: {t3 - t2:.4f}s")
        # Train directly on maximal_net with selected operations
        # This ensures gradients propagate back to SuperNet weights
        supernet.train()
        batch = next(iter(train_loader))
        inputs, targets = batch
        inputs, targets = inputs.to("cuda:0"), targets.to("cuda:0")
        optimizer.zero_grad()
        outputs = supernet.maximal_net(inputs, selected_ops=selected_ops)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        t4 = time.time()
        print(f"Training Time taken: {t4 - t3:.4f}s")
        dict_metrics["train_loss"].append(loss.item())
        print(f"Iteration {i+1}:  Train loss: {loss.item():.4f}. Architecture: {node.to_str()[:40]}...")

        # Write results to CSV
        csv_path = "supernet_result.csv"
        file_exists = os.path.isfile(csv_path)

        arch_str = node.to_str()
        small_net = supernet.sample(node)
        small_net.prune_cells()
        num_params = sum(p.numel() for p in small_net.parameters() if p.requires_grad)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['arch_str', 'train_loss', 'iteration', 'num_parameters'])
            writer.writerow([arch_str, loss.item(), i+1, num_params])
        

        #Free GPU memory
        torch.cuda.empty_cache()
        del node, small_net