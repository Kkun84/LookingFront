"""
Usage:
    train.py [--max_epochs=<int>] [--batch_size=<int>] [--image_size=<int>] [--lr=<float>] [-g=<int>...]

Options:
    -h --help           Show this screen.
    --max_epochs=<int>  Epoch num [default: 300].
    --batch_size=<int>  Batch size [default: 16].
    --image_size=<int>  Image width & height [default: 128].
    --lr=<float>        Learning rate.
    -g, --gpus=<int>    Use GPU [default: 0].
"""

import pytorch_lightning as pl
from docopt import docopt
from lightning_module import GAN, DataModule


def main():
    args = docopt(__doc__)

    print(args)

    max_epochs = int(args['--max_epochs'])
    batch_size = int(args['--batch_size'])
    image_size = int(args['--image_size'])
    lr = args['--lr'] and float(args['--lr'])
    gpus = [int(i) for i in args['--gpus']]

    datamodule = DataModule(batch_size=batch_size, image_size=image_size)
    model = GAN(**{k: v for k, v in dict(lr=lr).items() if v is not None})

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        progress_bar_refresh_rate=1,
    )
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
