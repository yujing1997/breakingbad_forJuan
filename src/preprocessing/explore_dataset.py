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
    
    set_root = "/Data/Seb/Hecktor2025/Task_1"
    assert os.path.exists(set_root), f"{set_root} not exist"
    cases = [x for x in glob.glob(f"{set_root}/*") if os.path.isdir(x)]
    MARGIN_MM = 20.0
    multiprocessing = True
    from_nnunet_raw = True
    def process_case(case_path, nnunet_raw_path:str="/Data/Seb/Hecktor2025/nnUNet_raw/Dataset_001_test", from_nnunet_raw:bool=from_nnunet_raw):
        print(f"Processing {case_path}")
        if from_nnunet_raw:
            case_id = os.path.basename(case_path).replace("-", "_")
            ct = sitk.ReadImage(os.path.join(nnunet_raw_path, "imagesTr", f"case_{case_id}_0001.nii.gz"))
            pet = sitk.ReadImage(os.path.join(nnunet_raw_path, "imagesTr", f"case_{case_id}_0002.nii.gz"))
            mask = sitk.ReadImage(os.path.join(nnunet_raw_path, "labelsTr", f"case_{case_id}.nii.gz"))
        else:
            ct, pet, mask = preprocess_case(case_path)
        ct_npy = sitk.GetArrayFromImage(ct)
        pet_npy = sitk.GetArrayFromImage(pet)
        max_HU_ct = np.max(ct_npy)
        min_HU_ct = np.min(ct_npy)
        mean_HU_ct = np.mean(ct_npy)
        median_HU_ct = np.median(ct_npy)
        max_SUV_pet = np.max(pet_npy)
        min_SUV_pet = np.min(pet_npy)
        mean_SUV_pet = np.mean(pet_npy)
        median_SUV_pet = np.median(pet_npy)
        return max_HU_ct, min_HU_ct, mean_HU_ct, median_HU_ct, max_SUV_pet, min_SUV_pet, mean_SUV_pet, median_SUV_pet, os.path.basename(case_path)

    
    max_HUs_cts = []
    min_HUs_cts = []
    mean_HUs_cts = []
    median_HUs_cts = []
    max_SUVs_pets = []
    min_SUVs_pets = []
    mean_SUVs_pets = []
    median_SUVs_pets = []
    case_ids = []
    if not multiprocessing:
        for case in tqdm.tqdm(cases, desc="Processing cases"):
            print("POrocessing case:", case)
            # if not "CHUM-059" in case:
            #     continue
            max_HU_ct, min_HU_ct, mean_HU_ct, median_HU_ct, max_SUV_pet, min_SUV_pet, mean_SUV_pet, median_SUV_pet, case_id = process_case(case)
            max_HUs_cts.append(max_HU_ct)
            min_HUs_cts.append(min_HU_ct)
            mean_HUs_cts.append(mean_HU_ct)
            median_HUs_cts.append(median_HU_ct)
            max_SUVs_pets.append(max_SUV_pet)
            min_SUVs_pets.append(min_SUV_pet)
            mean_SUVs_pets.append(mean_SUV_pet)
            median_SUVs_pets.append(median_SUV_pet)
            case_ids.append(case_id)
           
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            results = tqdm.tqdm(executor.map(process_case, cases), total=len(cases))
            for result in results:
                max_HU_ct, min_HU_ct, mean_HU_ct, median_HU_ct, max_SUV_pet, min_SUV_pet, mean_SUV_pet, median_SUV_pet, case_id = result
                max_HUs_cts.append(max_HU_ct)
                min_HUs_cts.append(min_HU_ct)
                mean_HUs_cts.append(mean_HU_ct)
                median_HUs_cts.append(median_HU_ct)
                max_SUVs_pets.append(max_SUV_pet)
                min_SUVs_pets.append(min_SUV_pet)
                mean_SUVs_pets.append(mean_SUV_pet)
                median_SUVs_pets.append(median_SUV_pet)
                case_ids.append(case_id)

    print("Patient with the smallest min HUs CT:",case_ids[np.argmin(min_HUs_cts)], "with value:", min_HUs_cts[np.argmin(min_HUs_cts)])
    print("Patient with the largest max HUs CT:", case_ids[np.argmax(max_HUs_cts)], "with value:", max_HUs_cts[np.argmax(max_HUs_cts)])
    print("Patient with the smallest mean HUs CT:",case_ids[np.argmin(mean_HUs_cts)], "with value:", mean_HUs_cts[np.argmin(mean_HUs_cts)])
    print("Patient with the largest mean HUs CT:",case_ids[np.argmax(mean_HUs_cts)], "with value:", mean_HUs_cts[np.argmax(mean_HUs_cts)])
    print("Patient with the smallest median HUs CT:", case_ids[np.argmin(median_HUs_cts)], "with value:", median_HUs_cts[np.argmin(median_HUs_cts)])
    print("Patient with the largest median HUs CT:", case_ids[np.argmax(median_HUs_cts)], "with value:", median_HUs_cts[np.argmax(median_HUs_cts)])
    print("Patient with the smallest min SUVs PET:", case_ids[np.argmin(min_SUVs_pets)], "with value:", min_SUVs_pets[np.argmin(min_SUVs_pets)])
    print("Patient with the largest max SUVs PET:", case_ids[np.argmax(max_SUVs_pets)], "with value:", max_SUVs_pets[np.argmax(max_SUVs_pets)])
    print("Patient with the smallest mean SUVs PET:", case_ids[np.argmin(mean_SUVs_pets)], "with value:", mean_SUVs_pets[np.argmin(mean_SUVs_pets)])
    print("Patient with the largest mean SUVs PET:", case_ids[np.argmax(mean_SUVs_pets)], "with value:", mean_SUVs_pets[np.argmax(mean_SUVs_pets)])
    print("Patient with the smallest median SUVs PET:", case_ids[np.argmin(median_SUVs_pets)], "with value:", median_SUVs_pets[np.argmin(median_SUVs_pets)])
    print("Patient with the largest median SUVs PET:", case_ids[np.argmax(median_SUVs_pets)], "with value:", median_SUVs_pets[np.argmax(median_SUVs_pets)])
   
    local_file_path = os.path.abspath( inspect.getfile(inspect.currentframe()))
    repo_src_path = Path(local_file_path).parents[1]

    if from_nnunet_raw:
        suffix = "preprocessed_nnunetraw"
    else:
        suffix = ""
    # PLotting distributions of max/min/mean/median HUs and SUVs
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.hist(max_HUs_cts, bins=50, color='blue', alpha=0.7)
    plt.title('Max HUs CT Distribution')
    plt.xlabel('Max HUs CT')
    plt.ylabel('Frequency')
    plt.subplot(2, 2, 2)
    plt.hist(min_HUs_cts, bins=50, color='orange', alpha=0.7)
    plt.title('Min HUs CT Distribution')
    plt.xlabel('Min HUs CT')
    plt.ylabel('Frequency')
    plt.subplot(2, 2, 3)
    plt.hist(mean_HUs_cts, bins=50, color='green', alpha=0.7)
    plt.title('Mean HUs CT Distribution')
    plt.xlabel('Mean HUs CT')
    plt.ylabel('Frequency')
    plt.subplot(2, 2, 4)
    plt.hist(median_HUs_cts, bins=50, color='red', alpha=0.7)
    plt.title('Median HUs CT Distribution')
    plt.xlabel('Median HUs CT')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(repo_src_path, "figures", f"max_min_mean_median_HU_CT_distributions_{suffix}.png"))
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.hist(max_SUVs_pets, bins=50, color='blue', alpha=0.7)
    plt.title('Max SUVs PET Distribution')
    plt.xlabel('Max SUVs PET')
    plt.ylabel('Frequency')
    plt.subplot(2, 2, 2)
    plt.hist(min_SUVs_pets, bins=50, color='orange', alpha=0.7)
    plt.title('Min SUVs PET Distribution')
    plt.xlabel('Min SUVs PET')
    plt.ylabel('Frequency')
    plt.subplot(2, 2, 3)
    plt.hist(mean_SUVs_pets, bins=50, color='green', alpha=0.7)
    plt.title('Mean SUVs PET Distribution')
    plt.xlabel('Mean SUVs PET')
    plt.ylabel('Frequency')
    plt.subplot(2, 2, 4)
    plt.hist(median_SUVs_pets, bins=50, color='red', alpha=0.7)
    plt.title('Median SUVs PET Distribution')
    plt.xlabel('Median SUVs PET')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(repo_src_path, "figures", f"max_min_mean_median_SUV_PET_distributions_{suffix}.png"))
   
