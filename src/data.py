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


# ── 전처리 레지스트리 (busbar 계승 — 결함 경계/대비 강조) ──────────────
import cv2


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _norm3(a):
    a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    return np.stack([a] * 3, -1).astype(np.uint8)


def pp_dog(img, s1=1.0, s2=2.5):
    """Difference of Gaussians — band-pass(결함 경계 강조). busbar 최강."""
    g = _to_gray(img)
    return _norm3(cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), s2))


def pp_retinex(img, sigma=30):
    """Single-scale Retinex — 조명 정규화."""
    g = _to_gray(img) + 1.0
    return _norm3(np.log(g) - np.log(cv2.GaussianBlur(g, (0, 0), sigma)))


def pp_clahe(img, clip=2.0):
    g = _to_gray(img).astype(np.uint8)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(g)
    return np.stack([c] * 3, -1).astype(np.uint8)


def pp_gamma(img, gamma=0.5):
    g = _to_gray(img) / 255.0
    return _norm3(np.power(g, gamma))


def pp_sharpen(img):
    """Unsharp mask — 경계 선명화."""
    g = _to_gray(img)
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    return _norm3(cv2.addWeighted(g, 1.5, blur, -0.5, 0))


def pp_clahe_dog(img):
    return pp_dog(pp_clahe(img))


def pp_dog_wide(img):
    return pp_dog(img, 2.0, 5.0)


def pp_dog_fine(img):
    return pp_dog(img, 0.5, 1.5)


def _kernel(k=9):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def pp_tophat(img, k=15):
    """White top-hat — 작은 '밝은' 결함 추출(점상)."""
    g = _to_gray(img)
    return _norm3(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, _kernel(k)))


def pp_blackhat(img, k=15):
    """Black-hat — 작은 '어두운' 결함 추출."""
    g = _to_gray(img)
    return _norm3(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, _kernel(k)))


def pp_sobel(img):
    g = _to_gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return _norm3(cv2.magnitude(gx, gy))


def pp_laplacian(img):
    return _norm3(np.abs(cv2.Laplacian(_to_gray(img), cv2.CV_32F, ksize=3)))


def pp_scharr(img):
    g = _to_gray(img)
    return _norm3(cv2.magnitude(cv2.Scharr(g, cv2.CV_32F, 1, 0),
                                cv2.Scharr(g, cv2.CV_32F, 0, 1)))


def pp_highpass(img, sigma=10):
    g = _to_gray(img)
    return _norm3(g - cv2.GaussianBlur(g, (0, 0), sigma))


def pp_bilateral_dog(img):
    """양방향 필터(노이즈 제거, 경계 보존) 후 DoG."""
    g = _to_gray(img).astype(np.uint8)
    b = cv2.bilateralFilter(g, 9, 75, 75).astype(np.float32)
    return _norm3(cv2.GaussianBlur(b, (0, 0), 1.0) - cv2.GaussianBlur(b, (0, 0), 2.5))


def pp_clahe_tophat(img):
    return pp_tophat(pp_clahe(img))


def pp_dog_xwide(img):
    return pp_dog(img, 3.0, 8.0)


def pp_log(img, sigma=2.0):
    """Laplacian of Gaussian — blob/ridge 검출."""
    g = cv2.GaussianBlur(_to_gray(img), (0, 0), sigma)
    return _norm3(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)))


def pp_gabor(img):
    """Gabor 방향 필터 뱅크 max(여러 방향의 줄무늬/스크래치)."""
    g = _to_gray(img)
    acc = None
    for th in np.arange(0, np.pi, np.pi / 4):
        k = cv2.getGaborKernel((21, 21), 4.0, th, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        r = np.abs(cv2.filter2D(g, cv2.CV_32F, k))
        acc = r if acc is None else np.maximum(acc, r)
    return _norm3(acc)


def pp_frangi(img):
    """Frangi ridge 필터(가늘고 긴 결함=스크래치 강조)."""
    from skimage.filters import frangi
    g = _to_gray(img) / 255.0
    return _norm3(frangi(g, sigmas=range(1, 4)))


def pp_eqhist(img):
    return np.stack([cv2.equalizeHist(_to_gray(img).astype(np.uint8))] * 3, -1).astype(np.uint8)


def pp_pctnorm(img, lo=2, hi=98):
    g = _to_gray(img)
    a, b = np.percentile(g, [lo, hi])
    return _norm3(np.clip(g, a, b))


def pp_zscore(img):
    g = _to_gray(img)
    return _norm3((g - g.mean()) / (g.std() + 1e-6))


def pp_gamma03(img):
    return pp_gamma(img, 0.3)


def pp_gamma15(img):
    return pp_gamma(img, 1.5)


def pp_clahe1(img):
    return pp_clahe(img, 1.0)


def pp_clahe4(img):
    return pp_clahe(img, 4.0)


def pp_morphgrad(img, k=5):
    g = _to_gray(img)
    return _norm3(cv2.morphologyEx(g, cv2.MORPH_GRADIENT, _kernel(k)))


def pp_canny(img):
    return np.stack([cv2.Canny(_to_gray(img).astype(np.uint8), 50, 150)] * 3, -1).astype(np.uint8)


def pp_tophat_big(img):
    return pp_tophat(img, 31)


PREPROCS = {
    "none": None,
    # 주파수/밴드패스/리지
    "dog": pp_dog, "dog_wide": pp_dog_wide, "dog_fine": pp_dog_fine, "dog_xwide": pp_dog_xwide,
    "log": pp_log, "highpass": pp_highpass, "bilateral_dog": pp_bilateral_dog,
    "gabor": pp_gabor, "frangi": pp_frangi,
    # 대비/조명
    "clahe": pp_clahe, "clahe1": pp_clahe1, "clahe4": pp_clahe4, "retinex": pp_retinex,
    "gamma": pp_gamma, "gamma03": pp_gamma03, "gamma15": pp_gamma15,
    "eqhist": pp_eqhist, "pctnorm": pp_pctnorm, "zscore": pp_zscore,
    # 형태학(작은 결함)
    "tophat": pp_tophat, "tophat_big": pp_tophat_big, "blackhat": pp_blackhat,
    "clahe_tophat": pp_clahe_tophat, "morphgrad": pp_morphgrad,
    # 에지/샤프
    "sobel": pp_sobel, "laplacian": pp_laplacian, "scharr": pp_scharr,
    "sharpen": pp_sharpen, "canny": pp_canny,
    # 조합
    "clahe_dog": pp_clahe_dog,
}


def build_tfms(train=True, preproc="none"):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    mean, std = (0.344, 0.344, 0.344), (0.18, 0.18, 0.18)  # 그레이 강판
    fn = PREPROCS.get(preproc)
    pre = [A.Lambda(image=lambda x, **k: fn(x), name=preproc)] if fn else []
    if train:
        return A.Compose(pre + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean, std),
            ToTensorV2(),
        ])
    return A.Compose(pre + [A.Normalize(mean, std), ToTensorV2()])
