import torch
from torch import Tensor, nn


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        start_chanels = 2 ** 9
        self.conv1 = nn.Conv2d(
            3 + 2, start_chanels * (2 ** 0), kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            start_chanels * (2 ** 0) + 2,
            start_chanels * (2 ** 0),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            start_chanels * (2 ** 0) + 2,
            start_chanels * (2 ** 0),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            start_chanels * (2 ** 0) + 2,
            start_chanels * (2 ** 0),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv5 = nn.Conv2d(
            start_chanels * (2 ** 0) + 2,
            start_chanels * (2 ** 0),
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(start_chanels * (2 ** 0), 1)

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def cat_coord(x: Tensor) -> Tensor:
        assert x.ndim == 4
        grid = torch.meshgrid(
            *[torch.linspace(-1, 1, s, device=x.device) for s in x.shape[2:4]]
        )
        coord = torch.stack(grid).unsqueeze(0).expand(len(x), -1, -1, -1)
        x = torch.cat([x, coord], 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.cat_coord(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.cat_coord(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.cat_coord(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.cat_coord(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.cat_coord(x)
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
    width, height = 128, 128

    summary(model, input_size=(batch_size, channel, height, width))
