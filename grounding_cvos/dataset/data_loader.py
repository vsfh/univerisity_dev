# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from shapely.geometry import Polygon

cv2.setNumThreads(0)

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
    print(bbox, flush=True)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
    )
    cv2.rectangle(
        img,
        (x_min, y_min - int(1.3 * text_height)),
        (x_min + text_width, y_min),
        BOX_COLOR,
        -1,
    )
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


class RSDataset(Dataset):
    def __init__(
        self,
        data_root,
        seg_root,
        data_name="CVOGL",
        split_name="train",
        img_size=1024,
        transform=None,
        augment=False,
    ):
        self.data_root = data_root
        self.seg_root = seg_root
        self.data_name = data_name
        self.img_size = img_size
        self.transform = transform
        self.split_name = split_name
        self.augment = augment

        if self.data_name == "CVOGL_DroneAerial":
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(
                data_dir, "{0}_{1}.pth".format(self.data_name, split_name)
            )
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, "query")
            self.rsimg_dir = os.path.join(data_dir, "satellite")
            self.rs_wh = self.img_size
            self.query_featuremap_hw = (256, 256)  # 52 #32
        elif self.data_name == "CVOGL_SVI":
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(
                data_dir, "{0}_{1}.pth".format(self.data_name, split_name)
            )
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, "query")
            self.rsimg_dir = os.path.join(data_dir, "satellite")
            self.rs_wh = self.img_size
            self.query_featuremap_hw = (256, 512)
        else:
            assert False

        self.rs_transform = albumentations.Compose(
            [
                albumentations.RandomSizedBBoxSafeCrop(
                    width=self.rs_wh, height=self.rs_wh, erosion_rate=0.2, p=0.4
                ),
                albumentations.OneOf(
                    [
                        albumentations.RandomRotate90(p=1),
                        albumentations.Rotate(limit=[180, 180], p=1),
                        albumentations.Rotate(limit=[270, 270], p=1),
                    ],
                    p=0.75,
                ),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
            ],
            bbox_params=albumentations.BboxParams(format="pascal_voc"),
            additional_targets={"mask": "mask"},
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name = self.data_list[
            idx
        ]

        ## box format: to x1y1x2y2
        bbox = np.array(bbox, dtype=int)

        bbox = bbox / (1024.0 / self.rs_wh)
        bbox = bbox.astype(int)

        queryimg = cv2.imread(os.path.join(self.queryimg_dir, queryimg_name))
        queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)

        rsimg = cv2.imread(os.path.join(self.rsimg_dir, rsimg_name))
        rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)

        mask_name = "{}--bbox({},{},{},{}).jpg".format(
            rsimg_name[:-4], int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        )
        mask_path = os.path.join(self.seg_root, "mask-satellite", mask_name)
        mask_rsimg = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_rsimg = (mask_rsimg > 127).astype(np.float32)

        if self.augment:
            rs_transformed = self.rs_transform(
                image=rsimg, bboxes=[list(bbox) + [cls_name]], mask=mask_rsimg
            )
            rsimg = rs_transformed["image"]
            bbox = rs_transformed["bboxes"][0][0:4]
            mask_rsimg = rs_transformed["mask"]

        # Norm, to tensor
        if self.transform is not None:
            rsimg = self.transform(rsimg.copy())
            queryimg = self.transform(queryimg.copy())

        query_featuremap_hw = self.query_featuremap_hw
        click_hw = (int(click_xy[1]), int(click_xy[0]))

        flag = random.choice([True, False])
        if self.split_name == "train" and flag:
            queryimg = torch.flip(queryimg, dims=[-1])
            click_hw = (click_hw[0], queryimg.shape[-1] - click_hw[1] - 1)

        mat_clickhw = np.zeros(
            (query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32
        )
        click_h = [pow(one - click_hw[0], 2) for one in range(query_featuremap_hw[0])]
        click_w = [pow(one - click_hw[1], 2) for one in range(query_featuremap_hw[1])]
        norm_hw = pow(
            query_featuremap_hw[0] * query_featuremap_hw[0]
            + query_featuremap_hw[1] * query_featuremap_hw[1],
            0.5,
        )
        for i in range(query_featuremap_hw[0]):
            for j in range(query_featuremap_hw[1]):
                tmp_val = 1 - (pow(click_h[i] + click_w[j], 0.5) / norm_hw)
                mat_clickhw[i, j] = tmp_val * tmp_val

        return (
            queryimg,
            rsimg,
            mat_clickhw,
            np.array(bbox, dtype=np.float32),
            mask_rsimg,
        )
