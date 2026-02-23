# BABY-AI: anisotropic to isotropic reconstruction of infant brain mri with deep learning

This repo contains the code to reconstruct anisotropic T2w infant brain MRI to isotropic. The repository can either be used to perform direct inference with a pretrained model, or to perform training on a new dataset. The project was initially conducted as a Master's Thesis at the Mathematical Faculty at Lund University by Alma Lennartsson.

Master's thesis available at: (link to lup). 

### Acknowledgements

- This model builds upon the [Residual U-Net](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unet.py) by MONAI.
- The data used to train the model is from [The Baby Open Brain's Repository](https://bobsrepository.readthedocs.io/) by <u>Feczko</u> et al.

## Structure

The repository is divided into inference and training. These can be used independently and for different purposes. 

**Inference**
This part relies on a pretrained model and can be directly used on new data. It only requires CPU power and is realtively fast performed. The user can input an ansiotropic dataset which the model reconstructs to an isotropic output saved in the `results/` directory

**Training**
This part is used to retrain the model to a new dataset. It requires CPU power and cuda, and has been performed on a HPC cluster during development. After training, the retrained model is saved to the `results/` directory and can be further used to perform inference and evaluations.

```bash
project/
├── README.md   # This file
├── scripts/
│   ├── functions.py                    # Help functions
│   ├── dataset.py                      # Implemenation of the dataloder
│   └── inference/
│   │   ├── demo.ipynb                  # An easy-to-follow Jupyter notebook which inputs one image and outputs the reconstructed version of it
│   │   ├── inference.py                # To reconstruct several imgaes from anisotropic to isotropic
│   │   └── requirements_inference.txt  # Requirements needed for all scripts under the inference directory
│   └── training/
│       ├── evaluation.py        # Evaluation functions and metrics computation
│       ├── init_run.py          # Initialization and data loading routines
│       ├── log_redirector.py    # Custom logging setup
│       ├── params.py            # Command-line argument parsing and settings
│       ├── results_traker.py    # Tracking and saving simulation results
│       ├── simulated_atrophy.py # Core simulation functions for atrophy and protein spread
│       ├── summary.py           # Functions to extract and summarize simulation metrics
│       └── utils.py             # Utility functions for plotting and other tasks
│   │   └── synthesize_anisotropic.py   # To generate more training data, creates anisotropic images from isotropic images
└── results/
```

## Files
All files are available via Huggingface and automatically loads in the scripts

## Installation

Note that the training scripts usually requires GPU
**Inference** 
1. Clone the repo or copy the files needed.
2. Set up a Python environment.
3. Install `requirements_inference.txt`.

**Training**  
1. Clone the repo or copy the files needed.
2. Set up a Python environment.
3. Install `requirements_training.txt`.


## Usage

### Preprocessing
hur amn gör det, skapa en fil som laddar upp atlas
- Ensure all images has size (182, 218, 182) mm and voxel size (1, 1, 1) mm
- Rigidly register all images to `sub-116056_ses-3mo_space-INFANTMNIacpc_T2w.nii.gz`

### Inference

### Training

