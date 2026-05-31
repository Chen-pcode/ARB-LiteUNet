import os
import sys
import warnings

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from arb_liteunet import ARBLiteUNet
from dataset_arb import SkinLesionDataset
from engine_arb import train_one_epoch, val_one_epoch, test_one_epoch
from config_arb_setting import setting_config
from utils import get_logger, get_optimizer, get_scheduler, log_config_info, set_seed

warnings.filterwarnings("ignore")


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
    main(setting_config)
