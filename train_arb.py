import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

from arb_liteunet import ARBLiteUNet
from arb_losses import ARBCompoundLoss
from dataset_arb import SkinLesionDataset
from engine_arb import train_one_epoch, val_one_epoch, test_one_epoch
from config_arb_setting import setting_config
from utils import (
    get_logger,
    get_optimizer,
    get_scheduler,
    log_config_info,
    myNormalize,
    myRandomHorizontalFlip,
    myRandomRotation,
    myRandomVerticalFlip,
    myResize,
    myToTensor,
    set_seed,
)

warnings.filterwarnings("ignore")


ABLATION_CHOICES = ("full", "no_dbs", "no_arcg", "no_brf")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARB-LiteUNet with reproducible ablation switches.")
    parser.add_argument("--ablation", type=str, default="full", choices=ABLATION_CHOICES)
    parser.add_argument("--datasets", type=str, default=None, choices=["isic17", "isic18", "ph2"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gpu_id", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default=None)
    return parser.parse_args()


def _dataset_path(data_root, datasets):
    mapping = {
        "isic17": "isic2017",
        "isic18": "isic2018",
        "ph2": "ph2",
    }
    return str(Path(data_root) / mapping.get(datasets, datasets)) + os.sep


def _build_transforms(config):
    config.train_transformer = transforms.Compose([
        myNormalize(config.datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(config.input_size_h, config.input_size_w),
    ])
    config.test_transformer = transforms.Compose([
        myNormalize(config.datasets, train=False),
        myToTensor(),
        myResize(config.input_size_h, config.input_size_w),
    ])


def apply_cli_config(config, args):
    if args.datasets is not None:
        config.datasets = args.datasets
    if args.data_root is not None:
        config.data_root = Path(args.data_root)
    config.data_path = _dataset_path(config.data_root, config.datasets)

    for attr in ("epochs", "seed", "batch_size", "num_workers", "gpu_id"):
        value = getattr(args, attr)
        if value is not None:
            setattr(config, attr, value)

    config.ablation = args.ablation
    config.model_config["use_arcg"] = args.ablation != "no_arcg"
    config.model_config["use_brf"] = args.ablation != "no_brf"

    if args.ablation == "no_dbs":
        config.criterion = ARBCompoundLoss(boundary_supervision_weight=0.0, surface_supervision_weight=0.0)
    else:
        config.criterion = ARBCompoundLoss()

    _build_transforms(config)

    if args.work_dir is not None:
        config.work_dir = args.work_dir.rstrip("/\\") + os.sep
    else:
        stamp = datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss")
        config.work_dir = os.path.join(
            "results",
            f"{config.network}_{config.datasets}_{args.ablation}_seed{config.seed}_{stamp}",
        ) + os.sep
    return config


def main(config):
    print("#----------Creating logger----------#")
    sys.path.append(config.work_dir + "/")
    log_dir = os.path.join(config.work_dir, "log")
    checkpoint_dir = os.path.join(config.work_dir, "checkpoints")
    outputs = os.path.join(config.work_dir, "outputs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    logger = get_logger("train", log_dir)
    writer = SummaryWriter(config.work_dir + "summary")
    log_config_info(config, logger)

    print("#----------GPU init----------#")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print("#----------Preparing dataset----------#")
    train_dataset = SkinLesionDataset(config.data_path, config, split="train")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    val_dataset = SkinLesionDataset(config.data_path, config, split="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=False)

    print("#----------Preparing Model----------#")
    model_cfg = config.model_config
    if config.network == "arb_liteunet":
        model = ARBLiteUNet(
            num_classes=model_cfg["num_classes"],
            input_channels=model_cfg["input_channels"],
            c_list=model_cfg["c_list"],
            use_arcg=model_cfg.get("use_arcg", True),
            use_brf=model_cfg.get("use_brf", True),
        )
    else:
        raise ValueError("network is not right")
    model = model.cuda()

    print("#----------Preparing loss, opt, sch----------#")
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    min_value = 999.0
    min_epoch = 1

    print("#----------Training----------#")
    for epoch in range(1, config.epochs + 1):
        torch.cuda.empty_cache()
        train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, writer)
        value = val_one_epoch(val_loader, model, criterion, epoch, logger, config)
        if value < min_value:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))
            min_value = value
            min_epoch = epoch

    best_path = os.path.join(checkpoint_dir, "best.pth")
    if os.path.exists(best_path):
        print("#----------Testing----------#")
        best_weight = torch.load(best_path, map_location=torch.device("cpu"))
        model.load_state_dict(best_weight)
        test_one_epoch(val_loader, model, criterion, logger, config, save_root=outputs)
        os.rename(best_path, os.path.join(checkpoint_dir, f"best-epoch{min_epoch}.pth"))


if __name__ == "__main__":
    main(apply_cli_config(setting_config, parse_args()))
