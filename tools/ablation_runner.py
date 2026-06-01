import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_CHOICES = ("full", "no_dbs", "no_arcg", "no_brf")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ARB-LiteUNet ablation training and optional unified evaluation.")
    parser.add_argument("--datasets", type=str, default="isic17", choices=["isic17", "isic18"])
    parser.add_argument("--modes", nargs="+", default=list(ABLATION_CHOICES), choices=ABLATION_CHOICES)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--eval", action="store_true", help="Run unified_eval.py after each training job.")
    parser.add_argument("--unified_eval_path", type=str, default=None)
    parser.add_argument("--isic_val_image_dir", type=str, default=None)
    parser.add_argument("--isic_val_mask_dir", type=str, default=None)
    parser.add_argument("--ph2_image_dir", type=str, default="./data/ph2/test/images")
    parser.add_argument("--ph2_mask_dir", type=str, default="./data/ph2/test/masks")
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_eval_script(path_arg):
    if path_arg:
        return Path(path_arg)
    candidates = [
        REPO_ROOT / "unified_eval.py",
        REPO_ROOT.parent / "unified_eval.py",
        REPO_ROOT.parents[2] / "result" / "unified_eval.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_command(cmd, dry_run=False):
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def dataset_val_dirs(dataset):
    name = "isic2017" if dataset == "isic17" else "isic2018"
    return f"./data/{name}/val/images", f"./data/{name}/val/masks"


def find_latest_checkpoint(dataset, mode, seed):
    pattern = REPO_ROOT / "results" / f"arb_liteunet_{dataset}_{mode}_seed{seed}_*" / "checkpoints" / "*.pth"
    candidates = [Path(p) for p in glob.glob(str(pattern))]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")

    def rank(path):
        name = path.name
        is_best_epoch = name.startswith("best-epoch")
        return (is_best_epoch, path.stat().st_mtime)

    return str(max(candidates, key=rank))


def build_train_cmd(args, mode):
    cmd = [
        sys.executable,
        "train_arb.py",
        "--datasets",
        args.datasets,
        "--ablation",
        mode,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--gpu_id",
        args.gpu_id,
    ]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.num_workers is not None:
        cmd += ["--num_workers", str(args.num_workers)]
    if args.data_root is not None:
        cmd += ["--data_root", args.data_root]
    return cmd


def build_eval_cmd(eval_script, model_weights, image_dir, mask_dir, dataset_name, norm_dataset, save_csv):
    return [
        sys.executable,
        str(eval_script),
        "--model",
        "arb",
        "--weights",
        model_weights,
        "--image_dir",
        image_dir,
        "--mask_dir",
        mask_dir,
        "--dataset_name",
        dataset_name,
        "--norm_dataset",
        norm_dataset,
        "--save_csv",
        save_csv,
    ]


def main():
    args = parse_args()
    eval_script = resolve_eval_script(args.unified_eval_path)
    if args.eval and not eval_script.exists():
        raise FileNotFoundError(
            "unified_eval.py was not found. Pass --unified_eval_path, for example "
            "../result/unified_eval.py from the project root layout."
        )

    isic_image_dir, isic_mask_dir = dataset_val_dirs(args.datasets)
    isic_image_dir = args.isic_val_image_dir or isic_image_dir
    isic_mask_dir = args.isic_val_mask_dir or isic_mask_dir
    save_csv = args.save_csv or f"ablation_{args.datasets}.csv"

    for mode in args.modes:
        run_command(build_train_cmd(args, mode), dry_run=args.dry_run)
        if not args.eval:
            continue

        checkpoint = find_latest_checkpoint(args.datasets, mode, args.seed)
        run_command(
            build_eval_cmd(
                eval_script,
                checkpoint,
                isic_image_dir,
                isic_mask_dir,
                f"{mode}_{args.datasets}_val",
                args.datasets,
                save_csv,
            ),
            dry_run=args.dry_run,
        )
        run_command(
            build_eval_cmd(
                eval_script,
                checkpoint,
                args.ph2_image_dir,
                args.ph2_mask_dir,
                f"{mode}_{args.datasets}_to_ph2",
                args.datasets,
                save_csv,
            ),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
