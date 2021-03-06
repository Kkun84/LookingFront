"""
Usage:
    realtime_predict.py <camera_id> <model_path> [--image_size=<int>] [-g=<int>]

Options:
    -h --help           Show this screen.
    --image_size=<int>  Image width & height [default: 128].
    -g, --gpus=<int>    Use GPU.
"""

from pathlib import Path

import cv2
from docopt import docopt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lightning_module import GAN, DataModule


def predict(input: np.ndarray, model, transform) -> np.ndarray:
    with torch.no_grad():
        x = Image.fromarray(input)
        x = transform(x)
        x = x[None].to(model.device)
        x = model(x)
        assert len(x) == 1
        x = transforms.functional.to_pil_image(x[0]).resize([min(input.shape[0:2])] * 2)
        x = np.asarray(x)
    return x


def main():
    args = docopt(__doc__)

    print(args)

    camera_id = int(args['<camera_id>'])
    model_path = Path(args['<model_path>'])
    image_size = int(args['--image_size'])
    gpus = args['--gpus'] and [int(i) for i in args['--gpus']]
    device = torch.device('cpu') if gpus is None else torch.device(f'cuda{gpus}')

    datamodule = DataModule(image_size=image_size)
    assert model_path.is_file()
    model = GAN.load_from_checkpoint(str(model_path)).to(device)
    model.eval()

    capture = cv2.VideoCapture(camera_id)
    try:
        cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Tile', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Compare', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = capture.read()
            assert ret
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Input', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            x = frame.copy()
            x[:, (1280 - 720) // 2 : -(1280 - 720) // 2] = predict(
                frame, model, datamodule.transform
            )
            cv2.imshow('Output', cv2.cvtColor(x, cv2.COLOR_RGB2BGR))

            cv2.imshow('Tile', cv2.cvtColor(cv2.vconcat([frame, x]), cv2.COLOR_RGB2BGR))
            cv2.imshow(
                'Compare', cv2.cvtColor((frame // 2 + x // 2), cv2.COLOR_RGB2BGR)
            )

            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print('Done.')
    return


if __name__ == '__main__':
    main()
