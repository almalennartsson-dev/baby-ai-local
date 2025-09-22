import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import decode_image
import nibabel as nib
import matplotlib.pyplot as plt


class TrainDataset(Dataset):
    def __init__(self, input_patches1, input_patches2, output_patches): # transform=None, target_transform=None):
        self.input_patches1 = input_patches1
        self.input_patches2 = input_patches2
        self.output_patches = output_patches

    def __len__(self):
        return len(self.output_patches)

    def __getitem__(self, idx):
        input_patch1 = self.input_patches1[idx]
        input_patch2 = self.input_patches2[idx]
        output_patch = self.output_patches[idx]

        # Optionally, convert to torch tensors
        input_patch1 = torch.from_numpy(input_patch1).float()
        input_patch2 = torch.from_numpy(input_patch2).float()
        output_patch = torch.from_numpy(output_patch).float()

        return input_patch1, input_patch2, output_patch