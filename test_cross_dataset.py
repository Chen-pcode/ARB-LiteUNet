import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from arb_liteunet import ARBLiteUNet
from dataset_arb import CrossDataset
from engine_arb import test_one_epoch
from config_arb_setting import setting_config
from arb_losses import ARBCompoundLoss
from utils import get_logger, log_config_info, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--dataset_name", type=str, default="cross_dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    config = setting_config
    config.gpu_id = args.gpu_id
    config.criterion = ARBCompoundLoss()

    sys.path.append(config.work_dir + "/")
    log_dir = os.path.join(config.work_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger("test", log_dir)
    log_config_info(config, logger)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)

    model = ARBLiteUNet(
        num_classes=config.model_config["num_classes"],
        input_channels=config.model_config["input_channels"],
        c_list=config.model_config["c_list"],
        use_arcg=config.model_config.get("use_arcg", True),
        use_brf=config.model_config.get("use_brf", True),
    ).cuda()
    weight = torch.load(args.weights, map_location=torch.device("cpu"))
    model.load_state_dict(weight)

    dataset = CrossDataset(args.image_dir, args.mask_dir, config)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers)
    test_one_epoch(loader, model, config.criterion, logger, config, test_data_name=args.dataset_name, save_root=os.path.join(config.work_dir, "outputs"))


if __name__ == "__main__":
    main()
