# BABY-AI: anisotropic to isotropic reconstruction of infant brain mri with deep learning

This repo contains code to reconstruct anisotropic T2w infant brain MRI to isotropic. The project was conducted as a Master's thesis available at: (link to lup). 

## Scripts
- `functions.py`: all functions
- `dataset.py`: dataset and patch extraction
- `demo.ipynb`: test the model by inputing one T1w isotropic MRI in combination with the T2w anisotropic MRI. Output is the reconstructed isotropic T2w MRI
- `synthesize_anisotropic.py`: create synthetic anisotropic images from isotropic images. Input the isotropic, choose slice thickness and direction, get the anisotropic image. 
- `training.py`: training script. Requires GPU.
- `batch_script.sh`: batch script for Berzelius HPC
- `random.ipynb`: my testing stuff, will clean this up.

## Files
- `weights_for_session2.pth`: model weights used in demo
- `bobsrepository`: training data from BOB's repository. Contains original isotropic data and synthesized anisotropic data.
- `sub-116056_ses-3mo_space-INFANTMNIacpc_T2w.nii.gz`: atals for rigid coregistration

## Installation
1. Clone the repo or copy the files needed.
2. Set up a Python environment (e.g., with venv or conda).
3. Install requirements.txt and requirements_cluster.txt

## Usage
1. Simple LR-HR reconstruction for a single anisotropic MRI - use `demo.ipynb`
2. Create augmented data, i.e anisotropic data from isotropic data - use `synthesize_anisotropic.py`
2. Train the model with additional data - use `training.py`

## Data Preparations
- Ensure all images has size (182, 218, 182) mm and voxel size (1, 1, 1) mm
- Rigidly register all images to `sub-116056_ses-3mo_space-INFANTMNIacpc_T2w.nii.gz`

## License
license?
