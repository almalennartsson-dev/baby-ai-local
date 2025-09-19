import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import decode_image
import nibabel as nib
from nibabel import processing
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.ndimage import zoom

def create_LR_img(img, scale_factor):
    """
    Downsamples a 3D image by the given scale factor. Interpolates to match original shape.
    
    Parameters:
    img (numpy array): The input 3D image to be LR.
    scale_factor (int): The factor by which to downsample the image.
    
    Returns:
    numpy array: The LR 3D image.
    """

    down_img = img[:, :, ::scale_factor]
    
    # Calculate zoom factors to match original shape
    zoom_factors = np.array(img.shape) / np.array(down_img.shape)

    # Upsample with interpolation
    up_img = zoom(down_img, zoom_factors, order=3)  # order=3: cubic interpolation

    return up_img


def scale_to_reference_img(img, ref_img):
    """
    Resamples img to match the shape and affine of ref_img using nibabel.
    
    Parameters:
    ref_img (nibabel NIfTI image): The reference image with desired shape and affine.
    img (nibabel NIfTI image): The image to be resampled.
    
    Returns:
    nibabel NIfTI image: The resampled image.
    """
    
    aligned_img = nib.processing.resample_from_to(img, ref_img)
    return aligned_img


def extract_3D_patches(img, patch_size, stride):
    """
    Extracts 3D patches from a 3D image.
    
    Parameters:
    img (numpy array): The input 3D image.
    patch_size (tuple): The size of each patch (depth, height, width).
    stride (tuple): The stride for patch extraction (depth_stride, height_stride, width_stride).
    
    Returns:
    list: A list of 3D patches.
    """
    
    patches = []
    d, h, w = img.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    for z in range(0, d - pd + 1, sd):
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                patch = img[z:z+pd, y:y+ph, x:x+pw] # can be clipped at edges
                patches.append(patch)
    
    return patches


def pad_to_shape(img, target_shape, img_affine):
    """
    Pads a 3D image to the target shape with zeros. Updates the affine matrix accordingly.
    
    Parameters:
    img (numpy array): The input 3D image.
    target_shape (tuple): The desired shape (depth, height, width).
    img_affine (numpy array): The original affine matrix of the image.
    
    Returns:
    numpy array: The padded 3D image.
    numpy array: The updated affine matrix.
    """
    pad_width = [( (t - s) // 2, (t - s) - (t - s) // 2 ) for s, t in zip(img.shape, target_shape)]
    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    
    new_affine = img_affine.copy()
    shift = np.array([pad[0] for pad in pad_width])
    new_affine[:3, 3] -= new_affine[:3, :3] @ shift

    return padded_img, new_affine #affine not adjusted!


def reconstruct_from_patches(patches, original_shape, stride):
    """
    Reconstructs the original 3D image from its patches.
    
    Parameters:
    patches (list): A list of 3D patches.
    original_shape (tuple): The shape of the original 3D image (depth, height, width).
    stride (tuple): The stride used during patch extraction (depth_stride, height_stride, width_stride).
    
    Returns:
    numpy array: The reconstructed 3D image.
    """
    
    reconstructed_img = np.zeros(original_shape)
    count_img = np.zeros(original_shape)  # To count overlaps
    
    pd, ph, pw = patches[0].shape
    sd, sh, sw = stride
    
    idx = 0
    for z in range(0, original_shape[0] - pd + 1, sd):
        for y in range(0, original_shape[1] - ph + 1, sh):
            for x in range(0, original_shape[2] - pw + 1, sw):
                reconstructed_img[z:z+pd, y:y+ph, x:x+pw] += patches[idx]
                count_img[z:z+pd, y:y+ph, x:x+pw] += 1
                idx += 1
    
    # Avoid division by zero
    count_img[count_img == 0] = 1
    reconstructed_img /= count_img
    
    return reconstructed_img


