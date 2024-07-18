from functools import partial

from coupler_detector.dataset import CouplerDataset

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torchvision.transforms import v2 as T

from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteHead, DefaultBoxGenerator
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []

    if train:
        pass

    transforms.extend([
        T.ToImage(),
        T.ToDtype(torch.float, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return T.Compose(transforms)


def filter_output(output, threshold, label_class):
    mask = (
        (output['labels'] == label_class)
        & (output['scores'] > threshold)
    )
    return {
        'boxes': output['boxes'][mask],
        'labels': output['labels'][mask],
        'scores': output['scores'][mask],
    }


def get_ssd320lite(num_classes=5, trainable_backbone_layers=6, pretrained=True, weights=None):
    '''
    trainable_backbone_layers 0-6 (6=all trainable backbone)
    '''
    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    num_anchors = anchor_generator.num_anchors_per_location()
    norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT or weights
    if not pretrained:
        weights = None
    model = ssdlite320_mobilenet_v3_large(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    model.head = SSDLiteHead(
        det_utils.retrieve_out_channels(model.backbone, size),
        num_anchors,
        num_classes,
        norm_layer,
    )

    return model


model = get_ssd320lite(num_classes=2, trainable_backbone_layers=6, pretrained=True)


def run_epoch(epoch_num=0):

    pbar = tqdm(total=len(train_ds))

    model.train()
    losses = []

    for b, (X, y) in enumerate(train_data_loader):
        optim.zero_grad()

        X = list(train_transforms(image) for image in X)

        output = model(X, y)

        loss = (output['bbox_regression'] + output['classification']) / len(X)

        if torch.isnan(loss):
            raise Exception('Loss exploded')

        loss.backward()
        optim.step()

        loss = loss.item()
        losses.append(loss)
        mean_loss = np.mean(losses)
        lr = scheduler.get_last_lr()
        pbar.set_description(f"{epoch_num=} {lr=} L:{mean_loss:0.4f}")
        pbar.update(len(X))

        del X

    # eval
    eval_losses = []

    for b, (X, y) in enumerate(test_data_loader):

        Xt = list(test_transforms(image) for image in X)
        model.train()
        output = model(Xt, y)
        loss = (output['bbox_regression'] + output['classification']) / len(X)

        eval_losses.append(loss.detach().numpy())

        model.eval()
        y = model(Xt, y)

        hist = np.histogram(y[0]['scores'].detach().numpy())
        # print(f"{hist=}")
        th = hist[1][5]

        y = filter_output(y[0], threshold=th, label_class=1)

        preview = T.ToImage()(X[0])
        preview = torchvision.utils.draw_bounding_boxes(preview, torch.tensor(y['boxes']), width=2)
        preview = np.array(T.ToPILImage()(preview))

        preview = cv2.resize(preview, [d // 2 for d in preview.shape[1::-1]])
        cv2.imwrite(f"preview_{b:02d}_e{epoch_num:02d}.jpg", np.array(preview))

        # cv2.imshow('test', np.array(preview))
        # cv2.waitKey(0)
        # cv2.destroyWindow('test')

    print('Train loss:', np.mean(losses), 'Eval loss:', np.mean(eval_losses))
    scheduler.step(np.mean(eval_losses))


if __name__ == "__main__":

    example_input = torch.rand(1, 3, 320, 320)
    model.eval()
    model(example_input)

    train_ds = CouplerDataset(train=True)
    test_ds = CouplerDataset(train=False)
    train_data_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

    optim = torch.optim.Adam(params=model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.3, patience=2, threshold=0.1)

    train_transforms = get_transform(train=True)
    test_transforms = get_transform(train=False)

    for i in range(15):
        run_epoch(epoch_num=i)

    fname = 'coupler_detector.torch'
    torch.save(model.state_dict(), fname)
    print(f"Saved {fname}")
