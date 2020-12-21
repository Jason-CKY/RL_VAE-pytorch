import glob
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from PIL import Image

from torch.nn import functional as F

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=transforms.ToTensor()):
        self.paths = glob.glob(data_dir+"/*.png")
        self.transforms = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transforms(image)

        return image, index

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_ratio = val_ratio

    def setup(self, stage):
        # transform
        transform=transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        # train/val split
        dataset = ImageDataset(self.data_dir, transform=transform)
        train_length = int(len(dataset)*self.val_ratio)
        train_dataset, val_dataset = random_split(dataset, [train_length, len(dataset)-train_length])

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
