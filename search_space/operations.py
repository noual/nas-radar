import torch
import torch.nn as nn

class Conv1x1(nn.Module):
    """1x1 convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x3_3x1(nn.Module):
    """1x3 followed by 3x1 convolution (separable-style)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Conv1x5_5x1(nn.Module):
    """1x5 followed by 5x1 convolution (separable-style)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DilatedConv3x3_r2(nn.Module):
    """3x3 dilated convolution with dilation rate 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DilatedConv3x3_r4(nn.Module):
    """3x3 dilated convolution with dilation rate 4"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Identity(nn.Module):
    """Identity operation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # If channels mismatch, use 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.match = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.match = None

    def forward(self, x):
        if self.match is not None:
            return self.match(x)
        return x


class Zero(nn.Module):
    """Zero operation (returns zeros)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        batch, _, height, width = x.size()
        return torch.zeros(batch, self.out_channels, height, width, device=x.device, dtype=x.dtype)
    

def get_op_candidates(in_channels, out_channels):
    """Returns a dictionary of operation candidates for LayerChoice"""
    return {
        'conv_1x1': Conv1x1(in_channels, out_channels),
        'conv_1x3_3x1': Conv1x3_3x1(in_channels, out_channels),
        'conv_1x5_5x1': Conv1x5_5x1(in_channels, out_channels),
        'dilated_conv_3x3_r2': DilatedConv3x3_r2(in_channels, out_channels),
        'dilated_conv_3x3_r4': DilatedConv3x3_r4(in_channels, out_channels),
        'identity': Identity(in_channels, out_channels),
        'none': Zero(in_channels, out_channels),
    }