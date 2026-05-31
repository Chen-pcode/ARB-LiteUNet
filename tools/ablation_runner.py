import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config_arb_setting import setting_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="full", choices=["full", "no_boundary", "no_artifact", "no_fusion"])
    args = parser.parse_args()

    config = setting_config
    if args.mode == "no_boundary":
        config.criterion.boundary_supervision_weight = 0.0
        config.model_config["use_brf"] = True
    elif args.mode == "no_artifact":
        config.model_config["use_arcg"] = False
    elif args.mode == "no_fusion":
        config.model_config["use_brf"] = False
    print(f"Ablation mode prepared: {args.mode}")


if __name__ == "__main__":
    main()
