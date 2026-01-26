import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import wandb
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.radar_dataset import RadarDavaDataset
from utils.models.unet import UNet

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

def dice_score(inputs, targets, smooth=1e-6):
    """Dice coefficient (1 - DiceLoss)."""
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float()
    targets = targets.float()
    intersection = (inputs * targets).sum()
    return float((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))

def iou_score(inputs, targets, smooth=1e-6):
    """IoU (Jaccard index)."""
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float()
    targets = targets.float()
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_dice += dice_score(outputs, targets)
            total_iou += iou_score(outputs, targets)
    n = len(dataloader)
    return total_loss / n, total_dice / n, total_iou / n


def train_unet(data_path, cell_str, epochs=200, batch_size=8, learning_rate=1e-5, name="model", in_features=16, problem_config=None):
    features = [in_features, in_features*2, in_features*4, in_features*8]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")

    # Dataset
    dataset = RadarDavaDataset(root_dir=data_path, batch_size=batch_size, has_distance=False)
    train_loader, val_loader, test_loader = dataset.generate_loaders()

    # Model
    if cell_str == "unet":
        model = UNet(in_channels=1, initial_channels=8, features=[16, 32, 64, 128]).to(device)
    else:
        node = RadarNode(problem_config)
        node.from_str(cell_str)
        supernet = SuperNet(in_channels=node.in_channels, initial_channels=node.initial_channels,
                            channel_options=node.channel_options, num_encoder_stages=node.num_encoder_stages,
                            num_nodes=node.num_nodes)
        model = supernet.sample(node).to(device)
        model.prune_cells()

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # W&B init
    wandb.init(project="nas-radar", name=name,
               config={"epochs": epochs, "batch_size": batch_size,
                       "learning_rate": learning_rate, "in_features": in_features,
                       "cell_str": cell_str},
               mode="online")
    wandb.watch(model, log="all", log_freq=100)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss, train_dice, train_iou = 0, 0, 0

        for step, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(outputs, targets)
            train_iou += iou_score(outputs, targets)

            # Log a sample prediction occasionally
            if np.random.randint(30) == 7:
                input_img = inputs[0].detach().cpu().numpy()
                pred_img = torch.sigmoid(outputs[0]).detach().cpu().numpy()
                label_img = targets[0].detach().cpu().numpy()

                wandb.log({
                    # "examples": [wandb.Image(input_img, caption="Input"),
                    #              wandb.Image(pred_img, caption="Prediction"),
                    #              wandb.Image(label_img, caption="Label")],
                    "train/loss_step": loss.item()
                })

        # Aggregate epoch results
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)

        wandb.log({"train/loss": train_loss,
                   "train/dice": train_dice,
                   "train/iou": train_iou,
                   "val/loss": val_loss,
                   "val/dice": val_dice,
                   "val/iou": val_iou,
                   "lr": optimizer.param_groups[0]["lr"],
                   "epoch": epoch})

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, "
              f"Train Dice {train_dice:.4f}, Val Dice {val_dice:.4f}, "
              f"Train IoU {train_iou:.4f}, Val IoU {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'results/best_{name}_model.pth')
            wandb.save(f'results/best_{name}_model.pth')

    return model

def main():
    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('--cell_str', type=str, required=True, help='Cell structure string')
    parser.add_argument('--in_features', type=int, default=16, help='Number of input features')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--name', type=str, default=datetime.now().strftime("%d_%H_%M"), help='Run name')
    parser.add_argument('--data_path', type=str, default="data/training_set/mat", help='Path to data')

    args = parser.parse_args()
    
    # Load config for problem settings
    from types import SimpleNamespace
    problem_config = SimpleNamespace(
        supernet=SimpleNamespace(
            in_channels=1,
            initial_channels=8,
            channel_options=[8, 16, 32, 64],
            num_encoder_stages=3,
            num_nodes=5
        )
    )
    
    model = train_unet(args.data_path, args.cell_str, epochs=args.epochs,
                       learning_rate=args.lr, batch_size=args.batch_size,
                       name=args.name, in_features=args.in_features, 
                       problem_config=problem_config)
    torch.save(model.state_dict(), f'./results/{args.name}_net.pth')
    print(f"Model saved as ./results/{args.name}_net.pth")

if __name__ == "__main__":
    main()