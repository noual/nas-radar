import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, initial_channels=8, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        # Initial convolution to map input to initial_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        in_channels = initial_channels

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                DoubleConv(feature*2, feature)
            ))
            # self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        skip_connections = []
        start_time = time.time()
        t0 = time.time()
        for i, down in enumerate(self.downs):
            x = down(x)
            # t2 = time.time()
            # print(f"Down {i}: {t2 - t0:.4f}")
            skip_connections.append(x)
            x = self.pool(x)
            # t3 = time.time()
            # t0 = t3

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # t4 = time.time()
        # print(f"Bottleneck: {t4 - t0:.4f}")

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)
            t5 = time.time()
            skip_conn = skip_connections[idx//2]

            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:])

            x = torch.cat((skip_conn, x), dim=1)
            # t6 = time.time()
            # print(f"Concat {idx}: {t6 - t5:.4f}")
            x = self.ups[idx+1](x)
            # t7 = time.time()
            # print(f"DoubleConv {idx}: {t7 - t6:.4f}")
        #     t4 = t7
        # print(f"Total forward time: {time.time() - start_time:.4f}")
        return self.final_conv(x)
