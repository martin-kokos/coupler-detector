import json
import random

import cv2
import numpy as np
from torchvision.datasets import VisionDataset
import torch
import structlog

import coupler_detector

log = structlog.getLogger()


class CouplerDataset(VisionDataset):

    def __init__(self, train=True, split=0.95, limit=999):
        self.dataset_path = coupler_detector.PROJECT_PATH / 'car_coupling_train'
        self.labels = []
        self.img_paths = []
        self.random = random.Random()
        self.random.seed(0)

        lbl_paths = sorted(list(self.dataset_path.glob('*.json')))
        self.random.shuffle(lbl_paths)

        for lbl_path in lbl_paths[:limit]:
            with open(lbl_path, 'rb') as f:
                data = json.load(f)
                img_path = self.dataset_path / data['imagePath']
                if img_path.exists():
                    assert len(data['shapes']) == 1

                    boxes = []
                    for shape in data['shapes']:
                        xmin = min([p[0] for p in shape['points']])
                        ymin = min([p[1] for p in shape['points']])
                        xmax = max([p[0] for p in shape['points']])
                        ymax = max([p[1] for p in shape['points']])
                        boxes.append([xmin, ymin, xmax, ymax])

                    self.labels.append({
                        'path': str(img_path),
                        'shapes': data['shapes'],
                        'boxes': torch.tensor(boxes),
                        'labels': torch.tensor([1] * len(boxes)),
                    })
                    self.img_paths.append(img_path)

                else:
                    log.warning(f'{img_path} doesn\'t exist')

        split_n = int(len(self.labels) * split)
        if train:
            self.labels = self.labels[:split_n]
            self.img_paths = self.img_paths[:split_n]
        else:
            self.labels = self.labels[split_n:]
            self.img_paths = self.img_paths[split_n:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lbl = self.labels[idx]
        img = cv2.imread(str(self.img_paths[idx]))  # BGR but we're B/W

        return img, lbl


if __name__ == "__main__":
    cd = CouplerDataset(train=False)

    print("Validation dataset:")
    for img, label in cd:
        print(label['path'])

        pts = np.array(label['shapes'][0]['points'], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), thickness=2)
        box = label['boxes'].to(dtype=torch.int32).squeeze().tolist()
        cv2.rectangle(img=img, pt1=box[:2], pt2=box[2:], color=(255, 0, 0), thickness=2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyWindow('img')
