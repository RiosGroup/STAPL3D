import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """"""

    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):

        return self.conv(x)


class UNET(nn.Module):
    """"""

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):

        super(UNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:

            self.downs.append(
                DoubleConv(
                    in_channels,
                    feature,
                    ),
                )
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):

            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                    )
                )

            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # Crop or add padding (Pooling and upsize by 2 removes pixels.
                # Say you have 161x161 will become 80x80 will become 160x160)
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='trilinear')

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)

        return x
