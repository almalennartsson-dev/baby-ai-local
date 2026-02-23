#IMPORTS

from scripts.functions import *
import nibabel as nib
from monai.networks.nets import UNet
import torch
import pathlib as p
from huggingface_hub import hf_hub_download

# SPECIFY PARAMETERS

patch_size = (32, 32, 32)
stride = (16, 16, 16)
target_shape = (192, 224, 192)

DATA_DIR = p.Path.home()/"data" # path to folder with data

model_weights = hf_hub_download(repo_id="almalennartsson/baby-ai", filename="2026-02-15T11:23:48.627180_model_weights.pth") #load model weights from huggingface

net = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=1,
    channels=(32, 64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2),
    num_res_units=10,
    norm=None,
)
net.load_state_dict(torch.load(model_weights, map_location="cpu"))

#LOAD DATA

# Path to input images
t1_files = sorted(DATA_DIR.rglob("*T1w.nii.gz")) # isotropic t1w images
t2_lr_files = sorted(DATA_DIR.rglob("*T2w.nii.gz")) # anisotropic t2w images

# reassure correct shape and voxel size
for i in range(len(t1_files)):
    assert nib.load(t1_files[i]).shape == nib.load(t2_lr_files[i]).shape == (182,218,182)
    assert nib.load(t1_files[i]).header.get_zooms() == nib.load(t2_lr_files[i]).header.get_zooms() == (1.0,1.0,1.0)

# TEST NETWORK

for i in range(len(t1_files)):
    t1_patches, affine = get_patches_single_img(t1_files[i], patch_size, stride, target_shape) #extract patches from t1w image
    t2_lr_patches, affine = get_patches_single_img(t2_lr_files[i], patch_size, stride, target_shape) #extract patches from t2w image
    all_outputs = []
    net.eval()
    with torch.no_grad():
        for i in range(len(t1_patches)):
            input1 = torch.tensor(t1_patches[i]).float()
            input2 = torch.tensor(t2_lr_patches[i]).float()
            inputs = torch.stack([input1, input2], dim=0) .unsqueeze(0)
            output = net(inputs)
            all_outputs.append(output.squeeze(0).squeeze(0).cpu().numpy()) 
        t2_reconstructed = reconstruct_from_patches(all_outputs, target_shape, stride)
    nib.save(nib.Nifti1Image(t2_reconstructed, affine), p.path.cwd()/"results"/t2_lr_files[i].name.replace("T2w","T2w_reconstructed")) #save output image in results folder
