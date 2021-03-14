from pytorch_lightning.core import datamodule
import pytorch_lightning as pl

from lightning_module import DataModule, GAN


datamodule = DataModule(batch_size=16)
model = GAN()

trainer = pl.Trainer(
    gpus=1,
    max_epochs=300,
    progress_bar_refresh_rate=1,
)
trainer.fit(model, datamodule)
