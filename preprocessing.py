
from huggingface_hub import hf_hub_download

local_dir = ... #path to folder where you want to save the atlas MRI
atlas_mri = hf_hub_download(repo_id="almalennartsson/baby-ai", filename="infant_atlas.nii.gz", local_dir=local_dir) #load atlas MRI from huggingface
