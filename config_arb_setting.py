import os
from datetime import datetime
from pathlib import Path

from torchvision import transforms

from arb_losses import ARBCompoundLoss
from utils import myNormalize, myToTensor, myRandomHorizontalFlip, myRandomVerticalFlip, myRandomRotation, myResize


class setting_config:
    repo_root = Path(__file__).resolve().parent
    data_root = Path(os.environ.get("SKIN_DATA_ROOT", repo_root / "data"))

    network = "arb_liteunet"
    model_config = {
        "num_classes": 1,
        "input_channels": 3,
        "c_list": [8, 12, 16, 32, 48, 64],
        "use_arcg": True,
        "use_brf": True,
    }

    datasets = "isic17"
    if datasets == "isic18":
        data_path = str(data_root / "isic2018") + os.sep
    elif datasets == "isic17":
        data_path = str(data_root / "isic2017") + os.sep
    elif datasets == "ph2":
        data_path = str(data_root / "ph2") + os.sep
    else:
        data_path = str(data_root / datasets) + os.sep

    criterion = ARBCompoundLoss()
    pretrained_path = "./pre_trained/"
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = "0"
    batch_size = 8
    epochs = 300

    work_dir = "results/" + network + "_" + datasets + "_" + datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss") + "/"

    print_interval = 20
    val_interval = 15
    save_interval = 5
    threshold = 0.5

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w),
    ])
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w),
    ])

    opt = "AdamW"
    if opt == "AdamW":
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    elif opt == "Adam":
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-4
        amsgrad = False
    else:
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-4
        amsgrad = False

    sch = "CosineAnnealingLR"
    if sch == "CosineAnnealingLR":
        T_max = 50
        eta_min = 1e-5
        last_epoch = -1
    elif sch == "StepLR":
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
