from functions import *
import pathlib as p
import nibabel as nib

# This setup creates anisotropic images with slice thickness 2, 3, 4, 5 mm in all directions (axial, sagittal, coronal) from the isotropic ground truth images in the data directory.

#LOAD ISOTROPIC GROUND TRUTH IMAGES
DATA_DIR = ... #path to folder with data
isotropic_images = sorted(DATA_DIR.rglob("*T2w.nii.gz")) # collect all isotropic t2w images in the data directory

#CREATE ANISOTROPIC IMAGES
scale_factors = [2, 3, 4, 5] # slice thickness in mm
start_slice = 0 # which slices to remove, 0 means even slices are removed, 1 means odd slices are removed
directions = ['axial', 'sagittal', 'coronal'] # directions can be 'axial', 'sagittal' or 'coronal' depending on which plane you want to remove slices from
LR_DIR = DATA_DIR/"LR_data"

for i in range(len(isotropic_images)):
    for scale_factor in scale_factors:
        for direction in directions:
            img = nib.load(isotropic_images[i])
            anisotropic_image = create_LR_img(img.get_fdata(), scale_factor, start_slice, direction)
            dir = LR_DIR / direction / f"LR{scale_factor}"
            dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(anisotropic_image, img.affine), dir / isotropic_images[i].name.replace("T2w",f"{direction}_T2w_LR{scale_factor}"))
            print(f"Created {direction} anisotropic image with slice thickness {scale_factor} mm for {isotropic_images[i].name}")
