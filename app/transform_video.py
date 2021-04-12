"""
Usage:
    realtime_predict.py <video_path> <model_path> [--image_size=<int>] [-g=<int>] [-v=<bool>]

Options:
    -h --help               Show this screen.
    --image_size=<int>      Image width & height [default: 256].
    -g, --gpus=<int>        Use GPU.
    -v, --verbose=<bool>    Display cv2 window [default: True].
"""

from pathlib import Path

import cv2
from docopt import docopt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

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

    video_path = Path(args['<video_path>'])
    model_path = Path(args['<model_path>'])
    image_size = int(args['--image_size'])
    gpus = args['--gpus'] and [int(i) for i in args['--gpus']]
    device = torch.device('cpu') if gpus is None else torch.device(f'cuda{gpus}')
    verbose = bool(args['--verbose'])

    datamodule = DataModule(image_size=image_size)
    assert model_path.is_file()
    model = GAN.load_from_checkpoint(str(model_path)).to(device)
    model.eval()

    capture = cv2.VideoCapture(str(video_path))

    video_writer_list = [
        cv2.VideoWriter(
            str(video_path.parent / (video_path.stem + '_' * i + video_path.suffix)),
            cv2.VideoWriter_fourcc(*'mp4v'),
            capture.get(cv2.CAP_PROP_FPS),
            (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )
        for i in range(1, 3)
    ]
    try:
        if verbose:
            cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Tile', cv2.WINDOW_NORMAL)

        frame_list = []
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame_list.append(frame)

        for frame in tqdm(frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            x = frame.copy()
            x[:, (1280 - 720) // 2 : -(1280 - 720) // 2] = predict(
                frame, model, datamodule.transform
            )

            video_writer_list[0].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_writer_list[1].write(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))

            if verbose:
                cv2.imshow('Input', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imshow('Output', cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
                cv2.imshow(
                    'Tile', cv2.cvtColor(cv2.vconcat([frame, x]), cv2.COLOR_RGB2BGR)
                )

            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        video_writer_list[0].release()
        video_writer_list[1].release()
        if verbose:
            cv2.destroyAllWindows()
        print('Done.')
    return


if __name__ == '__main__':
    main()
