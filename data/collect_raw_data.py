"""
Usage:
  collect_data.py <camera_id>
"""

from datetime import datetime
from pathlib import Path

import cv2
from docopt import docopt


def main():
    args = docopt(__doc__)

    camera_id = int(args['<camera_id>'])

    capture = cv2.VideoCapture(camera_id)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    save_path = Path('data', 'raw')

    label_list = ['front', 'around']
    print(label_list)

    save_path.mkdir(parents=True, exist_ok=False)
    for label in label_list:
        (save_path / label).mkdir(exist_ok=False)

    while True:
        now = datetime.now().strftime('%Y_%m_%d-%H_%M_%S-%f')
        print(now, end=' ')
        ret, frame = capture.read()
        assert ret

        cv2.imshow('title', frame)
        key = cv2.waitKey(0) & 0xFF
        if 0 <= key - ord('0') < len(label_list):
            label = label_list[key - ord('0')]
            print(label)
            filename = str(save_path / label / f'{now}.png')
            cv2.imwrite(filename, frame)
        else:
            print()
            if key == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    main()
