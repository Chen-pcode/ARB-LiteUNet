import argparse
import os
import time
import json
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import myNormalize, myToTensor, myResize, set_seed


class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, norm_dataset="isic17", h=256, w=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transformer = transforms.Compose([
            myNormalize(norm_dataset, train=False),
            myToTensor(),
            myResize(h, w),
        ])

        images = sorted(os.listdir(image_dir))
        masks = sorted(os.listdir(mask_dir))

        self.data = []
        for img_name, mask_name in zip(images, masks):
            self.data.append([
                os.path.join(image_dir, img_name),
                os.path.join(mask_dir, mask_name),
            ])

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.expand_dims(np.array(Image.open(mask_path).convert("L")), axis=2) / 255.0
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


def build_model(model_name, ablation="full"):
    c_list = [8, 12, 16, 32, 48, 64]

    if model_name == "ldeb":
        from ldebunet import LDEBUNet
        model = LDEBUNet(
            num_classes=1,
            input_channels=3,
            c_list=c_list,
        )
    elif model_name == "arb":
        from arb_liteunet import ARBLiteUNet
        use_arcg = ablation != "no_arcg"
        use_brf = ablation != "no_brf"
        model = ARBLiteUNet(
            num_classes=1,
            input_channels=3,
            c_list=c_list,
            use_arcg=use_arcg,
            use_brf=use_brf,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def get_final_mask(model_name, model_output):
    if model_name == "ldeb":
        # LDEB returns: gt_pre, key_points, out
        return model_output[2]

    if model_name == "arb":
        # ARB returns: deep_masks, deep_boundaries, final_mask
        return model_output[2]

    raise ValueError(model_name)


def calc_hd95(preds, gts):
    try:
        from medpy.metric.binary import hd95
    except Exception:
        return None

    vals = []
    for p, g in zip(preds, gts):
        p = p.astype(bool)
        g = g.astype(bool)
        if p.any() and g.any():
            vals.append(hd95(p, g))

    return float(np.mean(vals)) if vals else None


def profile_model(model, input_size=(1, 3, 256, 256)):
    params = sum(p.numel() for p in model.parameters())

    gflops = None
    try:
        from thop import profile
        dummy = torch.randn(*input_size).cuda()
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = macs / 1e9
    except Exception as exc:
        print(f"GFLOPs unavailable: {exc}")

    return params / 1e6, gflops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ldeb", "arb"])
    parser.add_argument("--ablation", default="full", choices=["full", "no_dbs", "no_arcg", "no_brf"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--dataset_name", default="test")
    parser.add_argument("--norm_dataset", default="isic17", choices=["isic17", "isic18", "isic"])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--save_csv", default="unified_eval_results.csv")
    parser.add_argument("--save_json", default="unified_eval_results.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_seed(42)

    dataset = SegDataset(
        args.image_dir,
        args.mask_dir,
        norm_dataset=args.norm_dataset,
        h=args.height,
        w=args.width,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model, args.ablation).cuda()
    weight = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(weight)
    model.eval()

    params_m, gflops = profile_model(model, input_size=(1, 3, args.height, args.width))

    pred_list = []
    gt_list = []
    image_mious = []
    model_times = []
    end_to_end_start = time.perf_counter()

    with torch.no_grad():
        for idx, (img, msk) in enumerate(tqdm(loader)):
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = model(img)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                output = model(img)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            out = get_final_mask(args.model, output)

            if idx >= args.warmup:
                model_times.append(elapsed_ms / 1000.0)

            out_np = out.squeeze(1).cpu().numpy()
            msk_np = msk.squeeze(1).cpu().numpy()

            pred_bin = (out_np >= args.threshold).astype(np.uint8)
            gt_bin = (msk_np >= 0.5).astype(np.uint8)

            inter = np.logical_and(pred_bin, gt_bin).sum()
            union = np.logical_or(pred_bin, gt_bin).sum()
            image_mious.append((inter + 1e-5) / (union + 1e-5))

            pred_list.append(pred_bin[0])
            gt_list.append(gt_bin[0])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_to_end_time = time.perf_counter() - end_to_end_start

    pred_flat = np.array(pred_list).reshape(-1)
    gt_flat = np.array(gt_list).reshape(-1)

    confusion = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    miou_global = TP / (TP + FP + FN) if (TP + FP + FN) else 0
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0
    acc = (TN + TP) / confusion.sum() if confusion.sum() else 0
    sen = TP / (TP + FN) if (TP + FN) else 0
    spe = TN / (TN + FP) if (TN + FP) else 0
    hd95_val = calc_hd95(np.array(pred_list), np.array(gt_list))

    avg_model_time = float(np.mean(model_times)) if model_times else None
    model_fps = 1.0 / avg_model_time if avg_model_time and avg_model_time > 0 else None
    end_to_end_fps = len(dataset) / end_to_end_time if end_to_end_time > 0 else None

    result = {
        "model": args.model,
        "ablation": args.ablation,
        "dataset_name": args.dataset_name,
        "image_count": len(dataset),
        "weights": args.weights,
        "image_dir": args.image_dir,
        "mask_dir": args.mask_dir,
        "norm_dataset": args.norm_dataset,
        "threshold": args.threshold,
        "params_m": params_m,
        "gflops": gflops,
        "miou_global": float(miou_global),
        "miou_image_mean": float(np.mean(image_mious)),
        "dice": float(dice),
        "accuracy": float(acc),
        "specificity": float(spe),
        "sensitivity": float(sen),
        "hd95": hd95_val,
        "model_time_ms_per_image": avg_model_time * 1000.0 if avg_model_time else None,
        "model_fps": model_fps,
        "end_to_end_time_s": end_to_end_time,
        "end_to_end_fps": end_to_end_fps,
        "confusion_matrix": confusion.tolist(),
    }

    print(json.dumps(result, indent=2))

    with open(args.save_json, "w") as f:
        json.dump(result, f, indent=2)

    file_exists = os.path.exists(args.save_csv)
    with open(args.save_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    main()
