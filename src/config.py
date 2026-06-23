"""공용 설정 — 경로/상수/시드."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TRAIN_IMG = os.path.join(DATA, "train_images")
TRAIN_CSV = os.path.join(DATA, "train.csv")
FOLDS_CSV = os.path.join(DATA, "folds.csv")
EXP = os.path.join(ROOT, "experiments")

H, W = 256, 1600
N_CLASSES = 4
SEED = 42

os.makedirs(EXP, exist_ok=True)
