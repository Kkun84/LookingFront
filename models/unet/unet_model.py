""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        start_chanels = 2 ** 5
        self.inc = DoubleConv(n_channels, start_chanels)
        self.down1 = Down(start_chanels, start_chanels * 2 ** 1)
        self.down2 = Down(start_chanels * 2 ** 1, start_chanels * 2 ** 2)
        self.down3 = Down(start_chanels * 2 ** 2, start_chanels * 2 ** 3)
        self.down4 = Down(start_chanels * 2 ** 3, start_chanels * 2 ** 4)
        self.down5 = Down(start_chanels * 2 ** 4, start_chanels * 2 ** 5 // factor)
        self.up1 = Up(
            start_chanels * 2 ** 5, start_chanels * 2 ** 4 // factor, bilinear
        )
        self.up2 = Up(
            start_chanels * 2 ** 4, start_chanels * 2 ** 3 // factor, bilinear
        )
        self.up3 = Up(
            start_chanels * 2 ** 3, start_chanels * 2 ** 2 // factor, bilinear
        )
        self.up4 = Up(
            start_chanels * 2 ** 2, start_chanels * 2 ** 1 // factor, bilinear
        )
        self.up5 = Up(start_chanels * 2 ** 1, start_chanels, bilinear)
        self.outc = OutConv(start_chanels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits
