import os 
import glob
from multiprocessing.pool import ThreadPool
import tqdm
import json

import numpy as np
import SimpleITK as sitk

from utils import preprocess_case, describe_array

# CT statistics
def get_HUs_and_pet_foreground_values(patient_paths, from_nnunet_raw:bool, nnunet_raw_path:str):  
    HU_values = []
    pet_values = []
    print(f"There are {len(patient_paths)} images.")
    def get_values_for_patient(patient_path, from_nnunet_raw=from_nnunet_raw, nnunet_raw_path=nnunet_raw_path):
        if from_nnunet_raw:
            case_id = os.path.basename(patient_path).replace("-", "_")
            ct = sitk.ReadImage(os.path.join(nnunet_raw_path, "imagesTr", f"case_{case_id}_0001.nii.gz"))
            pet = sitk.ReadImage(os.path.join(nnunet_raw_path, "imagesTr", f"case_{case_id}_0002.nii.gz"))
            mask = sitk.ReadImage(os.path.join(nnunet_raw_path, "labelsTr", f"case_{case_id}.nii.gz"))
        else:
            ct, pet, mask = preprocess_case(patient_path)
        ct_npy = sitk.GetArrayFromImage(ct)
        pet_npy = sitk.GetArrayFromImage(pet)
        mask_npy = sitk.GetArrayFromImage(mask)
        assert ct_npy.shape == pet_npy.shape == mask_npy.shape, \
            f"CT shape: {ct_npy.shape}, PET shape: {pet_npy.shape}, Mask shape: {mask_npy.shape} do not match for patient {patient_path}"
        HU_values.extend(ct_npy[mask_npy!=0].flatten())
        pet_values.extend(pet_npy[mask_npy!=0].flatten())

    # for patient_path in tqdm.tqdm(patient_paths, desc="Processing patients"):
    #     get_values_for_patient(patient_path)
    #     break
    # print(f"HU values: {len(HU_values)}, PET values: {len(pet_values)}")
    with ThreadPool(30) as pool:
        _ = list(tqdm.tqdm(pool.imap(get_values_for_patient, patient_paths), total=len(patient_paths)))

    return HU_values, pet_values



if __name__ == "__main__":

    import inspect
    import matplotlib.pyplot as plt
    data_path = "/Data/Seb/Hecktor2025/Task_1/"
    patient_paths = [folder for folder in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(folder)]
    from_nnunet_raw = True
    nnunet_raw_path = "/Data/Seb/Hecktor2025/nnUNet_raw/Dataset_001_test"
    HU_foreground_values, pet_foreground_values = get_HUs_and_pet_foreground_values(
        patient_paths, from_nnunet_raw=from_nnunet_raw, nnunet_raw_path=nnunet_raw_path)
    print("HU values mean: ", describe_array(HU_foreground_values))
    print("PET values mean: ", describe_array(pet_foreground_values))

    def format(data, dig:int=3):
        return str(round(data, dig))
    
    metadata = {
        "HU_foreground_values": {"mean": format(np.mean(HU_foreground_values)), "std": format(np.std(HU_foreground_values)), 
                                "median": format(np.median(HU_foreground_values)),
                                 "min": format(np.min(HU_foreground_values)), "max": format(np.max(HU_foreground_values))},
        "pet_foreground_values": {"mean": format(np.mean(pet_foreground_values)), "std": format(np.std(pet_foreground_values)),
                                    "median": format(np.median(pet_foreground_values)),
                                    "min": format(np.min(pet_foreground_values)), "max": format(np.max(pet_foreground_values))},
           }

    local_dir_name = os.path.dirname(os.path.abspath( inspect.getfile(inspect.currentframe())))
    statistics_file_path = os.path.join(os.path.dirname(local_dir_name), "constants", "statistics.json")
    with open(statistics_file_path, "w") as f:
        json.dump(metadata, f, indent=4)

    ## Plotting histograms of distributions for PET and CT foreground values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(HU_foreground_values, bins=100, color='blue', alpha=0.7)
    plt.title('Histogram of HU Foreground Values')
    plt.xlabel('HU Value')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(pet_foreground_values, bins=100, color='green', alpha=0.7)
    plt.title('Histogram of PET Foreground Values')
    plt.xlabel('PET Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(local_dir_name), "figures", "foreground_values_histograms.png"))




