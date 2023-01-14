
import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import torch

def open_image(path):
    return Image.open(path).convert("RGB")

class TargetDataset(data.Dataset):
    def __init__(self, dataset_folder):
        super().__init__()
        self.dataset_folder = dataset_folder
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512,512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.images_paths = sorted(glob(os.path.join(self.dataset_folder, "**", "*.jpg"), recursive=True))

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        #1 stands for night
        return normalized_img, 1 

    def __len__(self):
        return len(self.images_paths)


class DomainAdaptationDataLoader(data.DataLoader):
    def __init__(self, source_dataset, target_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_domain_iterator = data.DataLoader(source_dataset).__iter__()
        self.target_domain_iterator = data.DataLoader(target_dataset).__iter__()
        self.source_dim = int(kwargs["batch_size"] * 2 / 3)
        self.target_dim = kwargs["batch_size"] - self.source_dim
    def __iter__(self):
        return self

    def  __next__(self):
        try:
            source_batch = next(self.source_domain_iterator)
            target_batch = next(self.target_domain_iterator)
            batch = torch.cat((source_batch, target_batch),0)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

