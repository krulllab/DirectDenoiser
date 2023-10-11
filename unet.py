import torch
from torch import nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, activation=nn.ReLU()
    ):
        super().__init__()

        padding = kernel_size // 2

        block = []
        block.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="reflect",
                stride=stride,
            )
        )
        if activation is not None:
            block.append(activation)

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, n_convs=2):
        super().__init__()

        layers = [
            Conv(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        ] * n_convs
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class Merge(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()

        self.res_block = ResBlock(
            channels=channels * 2, kernel_size=kernel_size, n_convs=1
        )
        self.out_conv = Conv(
            in_channels=channels * 2, out_channels=channels, kernel_size=kernel_size
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.res_block(x)
        return self.out_conv(x)


class Resampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resample):
        super().__init__()

        if resample == "up":
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            conv = Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )
            self.layers = nn.Sequential(upsample, conv)

        if resample == "down":
            downsample = Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
            )
            self.layers = downsample

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, width):
        super().__init__()

        self.res_block = ResBlock(
            channels=in_channels, kernel_size=kernel_size, n_convs=width
        )
        self.resample = Resampling(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            resample="down",
        )

    def forward(self, x):
        skip = self.res_block(x)
        out = self.resample(skip)

        return out, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, width):
        super().__init__()
        
        self.res_block = ResBlock(
            channels=in_channels, kernel_size=kernel_size, n_convs=width
        )
        self.resample = Resampling(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            resample="up",
        )
        self.merge = Merge(channels=out_channels, kernel_size=kernel_size)

    def forward(self, x, skip):
        x = self.res_block(x)
        x = self.resample(x)
        out = self.merge(skip, x)

        return out


class UNet(nn.Module):
    def __init__(
        self,
        colour_channels=1,
        start_filters=8,
        kernel_size=3,
        depth=6,
        width=2,
        loss_fn="L2",
    ):
        super().__init__()
        self.loss_fn = loss_fn

        self.in_conv = Conv(
            in_channels=colour_channels,
            out_channels=start_filters,
            kernel_size=kernel_size,
        )

        encoder = []
        for i in range(depth):
            channels = start_filters * 2**i
            encoder.append(
                EncoderBlock(
                    in_channels=channels,
                    out_channels=channels * 2,
                    kernel_size=kernel_size,
                    width=width,
                )
            )
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for i in reversed(range(depth)):
            channels = start_filters * 2**i
            decoder.append(
                DecoderBlock(
                    in_channels=channels * 2,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    width=width,
                )
            )
        self.decoder = nn.ModuleList(decoder)

        out_block = []
        out_block.append(
            ResBlock(channels=start_filters, kernel_size=kernel_size, n_convs=width)
        )
        out_block.append(
            Conv(
                in_channels=start_filters,
                out_channels=colour_channels,
                kernel_size=kernel_size,
                activation=None,
            )
        )
        self.out_block = nn.Sequential(*out_block)

    def forward(self, x):
        x = self.in_conv(x)

        skips = []
        for layer in self.encoder:
            x, skip = layer(x)
            skips.append(skip)

        for layer in self.decoder:
            x = layer(x, skips[-1])
            skips.pop()

        out = self.out_block(x)

        return out

    def loss(self, x, y):
        if self.loss_fn == "L1":
            return F.l1_loss(x, y, reduction="none")
        elif self.loss_fn == "L2":
            return F.mse_loss(x, y, reduction="none")
