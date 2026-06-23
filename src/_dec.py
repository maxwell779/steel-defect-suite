import sys, torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from src import config, data
from src.analyze_seg import infer, decompose

if __name__ == "__main__":
    name = sys.argv[1]          # 실험 디렉토리명
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    arch = sys.argv[3] if len(sys.argv) > 3 else "unet"
    enc = sys.argv[4] if len(sys.argv) > 4 else "se_resnext50_32x4d"
    device = "cuda"
    ann = data.load_annotations(); _, va = data.split_fold(0)
    ld = DataLoader(data.SteelSegDataset(va, ann, data.build_tfms(False)),
                    batch_size=16, num_workers=4)
    A = {"unet": smp.Unet, "fpn": smp.FPN, "unetpp": smp.UnetPlusPlus}[arch]
    m = A(encoder_name=enc, encoder_weights=None, in_channels=3, classes=4).to(device).eval()
    m.load_state_dict(torch.load(f"{config.EXP}/{name}/best.pt", map_location=device))
    p, g, _ = infer(m, ld, device, thr=0.5, min_size=ms)
    print(f"=== {name} (min_size={ms}) ===")
    for d in decompose(p, g):
        print(f"  C{d['cls']}: 검출 {d['defect_recall']*100:.1f}% (FN{d['defect_FN']}/{d['n_def']}) "
              f"empty_FP{d['empty_FP']} 분할{d['seg_quality']:.3f} dice{d['mean_dice']:.3f}")
    print("DONE")
