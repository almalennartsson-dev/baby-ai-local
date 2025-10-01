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
    def __init__(self, input1, input2, output): # transform=None, target_transform=None):
        self.input1 = [patch for img_patches in input1 for patch in img_patches]
        self.input2 = [patch for img_patches in input2 for patch in img_patches]
        self.output = [patch for img_patches in output for patch in img_patches]

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        input1 = self.input1[idx]
        input2 = self.input2[idx]
        output = self.output[idx]

        input1 = torch.from_numpy(input1).float()
        input2 = torch.from_numpy(input2).float()
        output = torch.from_numpy(output).float()

        return input1, input2, output