
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Template dataset"""
    def __init__(self, data_root: str, split: str = "train", transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.image_paths = []
        # TODO: fill your image / label reading logic
        self.image_paths = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith((".jpg", ".png"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        # TODO: load ground‑truth label
        label = None

        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": label, "path": img_path}