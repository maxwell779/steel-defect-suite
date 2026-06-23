"""Severstal 데이터셋 — RLE→4채널 마스크, 이미지 단위 fold, albumentations 증강."""
import os, csv
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src import config

H, W, NC = config.H, config.W, config.N_CLASSES


def rle_decode(rle, h=H, w=W):
    """Severstal RLE(열 우선, 1-base) → (h,w) uint8 mask."""
    if not rle or rle != rle:  # empty/NaN
        return np.zeros((h, w), np.uint8)
    s = list(map(int, rle.split()))
    starts, lengths = s[0::2], s[1::2]
    m = np.zeros(h * w, np.uint8)
    for st, ln in zip(starts, lengths):
        m[st - 1: st - 1 + ln] = 1
    return m.reshape((w, h)).T  # 열 우선


def rle_encode(mask):
    """(h,w) bool/uint8 → Severstal RLE 문자열(열 우선,1-base). 빈마스크는 ''."""
    pix = mask.T.flatten()  # 열 우선
    pix = np.concatenate([[0], pix, [0]])
    runs = np.where(pix[1:] != pix[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs)) if len(runs) else ""


def load_annotations():
    """ImageId -> {classId(1-4): rle}."""
    ann = defaultdict(dict)
    with open(config.TRAIN_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ann[row["ImageId"]][int(row["ClassId"])] = row["EncodedPixels"]
    return ann


def load_folds():
    """(train_imgs, val_imgs) for a given fold + ImageId->fold."""
    rows = []
    with open(config.FOLDS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["ImageId"], int(row["fold"])))
    return rows


def split_fold(val_fold):
    rows = load_folds()
    tr = [i for i, fo in rows if fo != val_fold]
    va = [i for i, fo in rows if fo == val_fold]
    return tr, va


class SteelSegDataset(Dataset):
    """세그멘테이션: 이미지(3ch) + 4채널 마스크."""
    def __init__(self, image_ids, ann, tfm=None):
        self.ids = image_ids
        self.ann = ann
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        iid = self.ids[idx]
        img = np.array(Image.open(os.path.join(config.TRAIN_IMG, iid)).convert("RGB"))
        mask = np.zeros((H, W, NC), np.uint8)
        for c in range(1, NC + 1):
            if c in self.ann.get(iid, {}):
                mask[:, :, c - 1] = rle_decode(self.ann[iid][c])
        if self.tfm is not None:
            out = self.tfm(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
            # albumentations ToTensorV2: image CHW tensor, mask HWC tensor
            if not torch.is_tensor(img):
                img = torch.from_numpy(img.transpose(2, 0, 1))
            mask = mask.permute(2, 0, 1).float() if torch.is_tensor(mask) \
                else torch.from_numpy(mask.transpose(2, 0, 1)).float()
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return img, mask, iid


def build_tfms(train=True):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    mean, std = (0.344, 0.344, 0.344), (0.18, 0.18, 0.18)  # 그레이 강판
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean, std),
            ToTensorV2(),
        ])
    return A.Compose([A.Normalize(mean, std), ToTensorV2()])
