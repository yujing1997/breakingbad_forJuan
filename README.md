# Hecktor2025 Challenge at MICCAI

Following https://github.com/BioMedIA-MBZUAI/HECKTOR2025/ for Docker creation.


Most likely we will use a nnUnet pipeline from https://github.com/MIC-DKFZ/nnUNet/tree/master.


Create the nnUNet_raw dataset with preprocessing/raw_gen.py script.
```bash
python Hecktor2025/preprocessing/raw_gen.py -dp ./Data/Task\ 1/ -rn Dataset001_pet_zscore_noMD -nnunet_std
```

Preprocess from terminal with: 

```bash
export nnUNet_raw=path/to/your/nnUNet_raw
export nnUNet_preprocessed=path/to/your/nnUNet_preprocessed
export nnUNet_results=path/to/your/nnUNet_results
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
# Example: 
# export nnUNet_raw=/Data/Seb/Hecktor2025/nnUNet_raw
# export nnUNet_preprocessed=/Data/Seb/Hecktor2025/nnUNet_preprocessed
# export nnUNet_results=/Data/Seb/Hecktor2025/nnUNet_results
# nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

Then if you want to modify the nnUNet config you can add a config to the generated nnUNetPlans.json

```json
,
"3d_fullres_bs8": {
    "inherits_from": "3d_fullres",
    "batch_size": 8
}
```


To create the env on Narval
```bash
module load scipy-stack
# All default modules loaded are good (python3.10.13, StdEnv/2023, gcc/12.3)
module save hecktorModules
virtualenv --no-download hecktorenv
source hecktorenv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy
pip install --no-index simpleitk
pip install --no-index tqdm
cd nnUNet_HN
pip install -e .
```
