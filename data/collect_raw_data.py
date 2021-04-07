"""
Usage:
  collect_data.py <camera_id> [--exist_ok]

Options:
    -h, --help      Show this message and exit.
"""


from datetime import datetime
from pathlib import Path

import cv2
from docopt import docopt
from tqdm import tqdm


def main():
    args = docopt(__doc__)

    camera_id = int(args['<camera_id>'])
    exist_ok = bool(args['--exist_ok'])

    winname = 'collect_raw_data.py'
    capture = cv2.VideoCapture(camera_id)
    try:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        ret, frame = capture.read()
        assert ret
        cv2.imshow(winname, frame)

        save_path = Path('data', 'raw')

        label_list = ['front', 'around']

        save_path.mkdir(parents=True, exist_ok=exist_ok)
        for i, label in enumerate(label_list):
            (save_path / f'{i}_{label}').mkdir(exist_ok=exist_ok)

        while True:
            print('-' * (3 + max([len(i) for i in label_list])))
            print('\n'.join([f'{i}: {v}' for i, v in enumerate(label_list)]))
            print('-' * (3 + max([len(i) for i in label_list])))

            ret, frame = capture.read()
            cv2.imshow(winname, frame)
            key = cv2.waitKey(0) & 0xFF
            label_index = key - ord('0')
            if 0 <= label_index < len(label_list):
                label = label_list[label_index]
                print(label)
                flame_list = {}
                for i in tqdm(range(1000)):
                    now = datetime.now().strftime('%Y_%m_%d-%H_%M_%S-%f')
                    ret, frame = capture.read()
                    assert ret
                    flame_list[now] = frame
                    if i % 10 == 0:
                        cv2.imshow(winname, frame)
                        cv2.waitKey(1)
                for i, (now, frame) in enumerate(tqdm(flame_list.items())):
                    filename = str(save_path / f'{label_index}_{label}' / f'{now}.png')
                    if i % 10 == 0:
                        cv2.imshow(winname, frame)
                        cv2.waitKey(1)
                    cv2.imwrite(filename, frame)
            else:
                if key == ord('q'):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()
