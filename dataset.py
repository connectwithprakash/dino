import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class DinoDataset(Dataset):
    def __init__(self, file_path: str, transform=None):
        with open(file_path, "r") as f:
            representative_image_map = json.load(f)
        self.images = list(representative_image_map.values())
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, np.nan

if __name__ == "__main__":
    dataset = DinoDataset("temp_delete.json")
    print(dataset[0])
