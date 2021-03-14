from typing import Union
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size: int,
        data_path: Union[str, Path] = 'data/raw',
        batch_size: int = 64,
        num_workers: int = 24,
    ):
        super().__init__()
        self.data_path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(720),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train = ImageFolder(str(self.data_path / ''), transform=self.transform)
        self.valid = ImageFolder(str(self.data_path / ''), transform=self.transform)
        self.test = ImageFolder(str(self.data_path / ''), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
