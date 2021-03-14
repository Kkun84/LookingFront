import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)

        x = self.gap(x.relu())
        x = self.flatten(x)

        x = self.linear(x)

        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    model = Discriminator()

    batch_size = 2
    channel = 3
    width, height = 256, 32

    summary(model, input_size=(batch_size, channel, height, width))
