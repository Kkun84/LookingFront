import itertools
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision

from models import Generator, Discriminator


class GAN(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = Generator()
        self.discriminator = Discriminator()

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def forward(self, image_batch: Tensor):
        return self.generator(image_batch)

    def adversarial_loss(self, y_hat: Tensor, y: Tensor):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        image_batch, label = batch
        batchsize = len(image_batch)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_image_batch = self(image_batch)

            # log sampled images
            if batch_idx == 0:
                grid = torchvision.utils.make_grid(
                    list(
                        itertools.chain.from_iterable(
                            list(zip(image_batch, self.generated_image_batch))
                        )
                    ),
                    2,
                    pad_value=0.5,
                )
                self.logger.experiment.add_image(
                    f'generated_images #{self.current_epoch}', grid
                )

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(
                batchsize, 1, device=image_batch.device, dtype=torch.float32
            )

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(
                self.discriminator(self.generated_image_batch), valid
            )
            tqdm_dict = {'g_loss': g_loss}
            output = {'loss': g_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict}
        # train discriminator
        elif optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            mask = label == self.train_dataloader().dataset.class_to_idx['front']
            valid = torch.ones(
                batchsize, 1, device=image_batch.device, dtype=torch.float32
            )
            valid = valid * mask[:, None]
            real_loss = self.adversarial_loss(self.discriminator(image_batch), valid)

            # how well can it label as fake?
            fake = torch.zeros(
                batchsize, 1, device=image_batch.device, dtype=torch.float32
            )
            fake_loss = self.adversarial_loss(
                self.discriminator(self(image_batch).detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = {'loss': d_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict}
        else:
            assert False
        return output

    # def on_epoch_end(self):
    #     # log sampled images
    #     sample_image_batch = self(z)
    #     grid = torchvision.utils.make_grid(sample_image_batch)
    #     self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
