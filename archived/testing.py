#TEST ONE IMAGE
import pathlib as p
from scripts.functions import *
import nibabel as nib
import numpy as np
import random
import torch
from monai.networks.nets import UNet
from monai.networks.layers.factories import Norm
import matplotlib.pyplot as plt

#Parameters
patch_size = (32,32,32)
stride = (16,16,16)
target_shape = (192,224,192)

#Load files
DATA_DIR = p.Path.home()/"data"/"bobsrepository"
axial = DATA_DIR/"LR_data"/"axial"/"even"
t1_files = sorted(DATA_DIR.rglob("*T1w.nii.gz"))
t2_files = sorted(DATA_DIR.rglob("*T2w.nii.gz"))
t2_LR_files = sorted(axial.rglob("*T2w_LR.nii.gz"))

#reassure correct shape and voxel size
assert nib.load(t1_files[0]).shape == nib.load(t2_files[0]).shape == nib.load(t2_LR_files[0]).shape == (182,218,182)
assert nib.load(t1_files[0]).header.get_zooms() == nib.load(t2_files[0]).header.get_zooms() == nib.load(t2_LR_files[0]).header.get_zooms() == (1.0,1.0,1.0)

t1_patches = get_patches_single_img(t1_files[0], patch_size, stride, target_shape)
t2_lr_patches = get_patches_single_img(t2_LR_files[0], patch_size, stride, target_shape)


#Test deeper unet

net = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=1,
    channels=(32, 64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2),
    num_res_units=10, 
    norm=None,
)
net.load_state_dict(torch.load(DATA_DIR/"outputs"/"2025-12-10T15:25:46.850860_model_weights.pth", map_location="cpu"))

all_outputs = []
net.eval()
with torch.no_grad():
    for i in range(len(t1_patches)-1):
        input1 = torch.tensor(t1_patches[i]).float()
        input2 = torch.tensor(t2_lr_patches[i]).float()
        inputs = torch.stack([input1, input2], dim=0).unsqueeze(0)  # (1, 2, 32, 32, 32)
        output = net(inputs)
        all_outputs.append(output.squeeze(0).squeeze(0).cpu().numpy())  # (32, 32, 32)
    gen_reconstructed = reconstruct_from_patches(all_outputs, target_shape, stride)

nib.save(nib.Nifti1Image(gen_reconstructed, affine=pad_to_shape(nib.load(t1_files[0]), target_shape).affine), DATA_DIR/"images"/"session1_test1_deeperunet.nii.gz")
