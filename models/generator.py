from torch import nn
from models.unet import UNet


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = UNet(3, 3)
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        # self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.conv3 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        # x = self.conv1(x).relu()
        # x = self.conv2(x).relu()
        # x = self.conv3(x).sigmoid()
        return x


if __name__ == '__main__':
    from torchinfo import summary

    model = Generator()

    batch_size = 2
    channel = 3
    width, height = 128, 128

    summary(model, input_size=(batch_size, channel, height, width))
