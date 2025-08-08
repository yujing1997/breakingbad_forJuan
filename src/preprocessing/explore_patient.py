import os
import glob
import tqdm
import json
import inspect
from pathlib import Path
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from utils import preprocess_case


if __name__ == "__main__":
    
    set_root = "/home/sebq/HECKTOR2025/Data/Task_1"
    assert os.path.exists(set_root), f"{set_root} not exist"
    cases = [x for x in glob.glob(f"{set_root}/*") if os.path.isdir(x)]
    MARGIN_MM = 20.0
    multiprocessing = True

    def plot_case(case_path):
        print(f"Processing {case_path}")
        ct, pet, mask = preprocess_case(case_path)
        ct_npy = sitk.GetArrayFromImage(ct)
        pet_npy = sitk.GetArrayFromImage(pet)
        ct_body = ct_npy[(ct_npy>-500) & (ct_npy<1000)]
        pet_body = pet_npy[(ct_npy>-500) & (ct_npy<1000)]
        percentile_pet_80 = np.round(np.percentile(pet_body, 80), 2)
        percentile_pet_90 = np.round(np.percentile(pet_body, 90), 2)
        percentile_pet_95 = np.round(np.percentile(pet_body, 95), 2)
        percentile_pet_99 = np.round(np.percentile(pet_body, 99), 2)
        print(f"Percentiles for PET: 80%: {percentile_pet_80}, 90%: {percentile_pet_90}, 95%: {percentile_pet_95}, 99%: {percentile_pet_99}")
        # PLotting the distribution of the pet and of the CT
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(ct_body.flatten(), bins=100, color='blue', alpha=0.7)
        plt.title('CT Intensity Distribution')
        plt.xlabel('HU')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        plt.hist(pet_body.flatten(), bins=100, color='red', alpha=0.7)
        plt.title(f'PET Intensity Distribution, 80%: {percentile_pet_80}, 90%: {percentile_pet_90}, 95%: {percentile_pet_95}, 99%: {percentile_pet_99}')
        plt.xlabel('SUV')
        plt.ylabel('Frequency')
        plt.tight_layout()
        local_file_path = os.path.abspath( inspect.getfile(inspect.currentframe()))
        repo_src_path = Path(local_file_path).parents[1]
        plt.savefig(repo_src_path / "figures" / f"{os.path.basename(case_path)}_intensity_distribution.png")



    for case in tqdm.tqdm(cases, desc="Processing cases"):
        print("POrocessing case:", case)
        plot_case(case)
       



   