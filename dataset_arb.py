from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class SkinLesionDataset(Dataset):
    def __init__(self, path_data, config, split="train"):
        super().__init__()
        self.split = split
        self.transformer = config.train_transformer if split == "train" else config.test_transformer

        image_dir = os.path.join(path_data, split, "images")
        mask_dir = os.path.join(path_data, split, "masks")
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Missing dataset split directories: {image_dir} or {mask_dir}")

        images_list = sorted(os.listdir(image_dir))
        masks_list = sorted(os.listdir(mask_dir))
        self.data = []
        for img_name, mask_name in zip(images_list, masks_list):
            self.data.append([os.path.join(image_dir, img_name), os.path.join(mask_dir, mask_name)])

    def __getitem__(self, index):
        img_path, msk_path = self.data[index]
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert("L")), axis=2) / 255.0
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


class CrossDataset(Dataset):
    def __init__(self, image_dir, mask_dir, config):
        super().__init__()
        self.transformer = config.test_transformer
        images_list = sorted(os.listdir(image_dir))
        masks_list = sorted(os.listdir(mask_dir))
        self.data = []
        for img_name, mask_name in zip(images_list, masks_list):
            self.data.append([os.path.join(image_dir, img_name), os.path.join(mask_dir, mask_name)])

    def __getitem__(self, index):
        img_path, msk_path = self.data[index]
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert("L")), axis=2) / 255.0
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
