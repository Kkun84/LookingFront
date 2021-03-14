"""
Usage:
  train.py [--max_epochs=<int>] [--batch_size=<int>] [--image_size=<int>]

Options:
  -h --help           Show this screen.
  --max_epochs=<int>  Epoch num [default: 300].
  --batch_size=<int>  Batch size [default: 16].
  --image_size=<int>  Image width & height [default: 128].
"""

from docopt import docopt

import pytorch_lightning as pl

from lightning_module import DataModule, GAN


def main():
    args = docopt(__doc__)

    print(args)

    max_epochs = args['--max_epochs']
    batch_size = args['--batch_size']
    image_size = args['--image_size']

    datamodule = DataModule(batch_size=batch_size, image_size=image_size)
    model = GAN()

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        progress_bar_refresh_rate=1,
    )
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
