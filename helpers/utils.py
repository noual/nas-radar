import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

def configure_seaborn(**kwargs):
    sns.set_context("notebook")
    grid = kwargs.get("grid", False)
    sns.set_theme(sns.plotting_context("notebook", font_scale=1), style="whitegrid",
                  rc={
            # grid activation
            "axes.grid": False,
            # grid appearance
            "grid.color": "#BFBFBF",
            "axes.edgecolor": "#BFBFBF",
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            'font.family':'sans-serif',
            'font.sans-serif':['Lato'],
        },)

    palette = ["#3D405B", "#E08042", "#54AB69", "#CE2F49", "#A26EBF", "#75523A", "#D12AA2", "#E0D06F", "#6F9AA7", "#3359C4",
               "#76455B"]
    
    sns.set_palette(palette)