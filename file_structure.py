
import os
import random
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib

def save_images(file_list, output_dir):
    """
    Saves images as NIfTI files from the file list to the specified output directory.

    Parameters:
    - file_list: List of file paths.
    - output_dir: Directory where images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_path in file_list:
        img = nib.load(file_path)
        img_data = img.get_fdata()
        base_name = os.path.basename(file_path)
        save_path = os.path.join(output_dir, base_name)
        nib.save(nib.Nifti1Image(img_data, img.affine), save_path)

def extract_and_save_patches(img, patch_size, stride, output_dir, base_name):
    """ Extracts 3D patches from a 3D image and saves them as NIfTI files.

    Parameters:
    - img: The input 3D image (as a numpy array).
    - patch_size: The size of each patch (depth, height, width).
    - stride: The stride for patch extraction (depth_stride, height_stride, width_stride).
    - output_dir: Directory where patches will be saved.
    - base_name: Base name for the saved patch files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    patches = extract_3D_patches(img, patch_size, stride)
    save
    
    
