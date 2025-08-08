import argparse
import os
import json
import shutil
import glob
from tqdm import tqdm
from multiprocessing.pool import Pool
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from constants.labels import ids
from utils import preprocess_case
from cropping import crop_z_axis, crop_below_brain

import numpy as np
import SimpleITK as sitk


class RawGenerator:
    def __init__(self, data_path:str, raw_name:str, nnunet_std:bool=True):
        self.data_path = data_path
        self.raw_name = raw_name
        self.nnunet_std = nnunet_std
        self.nnunet_raw_path = os.environ["nnUNet_raw"]
        self.extension = ".nii.gz"
        self.number_of_cases = None
        # assert not os.path.exists(f'{self.nnunet_raw_path}/{self.raw_name}'), "Raw dataset already exists"
        if os.path.exists(f'{self.nnunet_raw_path}/{self.raw_name}'):
            answer = input(f'Dataset {raw_name} already exists, do you want to overwrite it?\n')
            if str(answer).lower() in ["t", "true", "y", "yes"]:
                shutil.rmtree(f'{self.nnunet_raw_path}/{self.raw_name}')
                os.makedirs(f'{self.nnunet_raw_path}/{self.raw_name}')
            else:
                print("You did not wish to continue. Getting out of the script")
                exit(0)
        else:
            os.makedirs(f'{self.nnunet_raw_path}/{self.raw_name}')

        
    def create_dataset_json(self):

        channel_name = {} 

        if self.nnunet_std:
            pet_std = "zscore"
        else:
            # Do we want to stdandardize based on foreground values of PET?
            pet_std = "ct"

        # First channel is always CT
        channel_name["0"] = "ct"
        # Second channel is always PET
        channel_name["1"] = pet_std
       
        labels = ids
    
        dataset_json = {
            "channel_names": channel_name,
            "labels": labels,
            "numTraining": self.number_of_cases,
            "file_ending": self.extension
        }

        with open(os.path.join(self.nnunet_raw_path, self.raw_name, "dataset.json"), "w") as file:
            json.dump(dataset_json, file, indent=4)
    

    def preprocess_data_single_case(self, patient_path:str, margin_mm_side_brain:float=50., margin_mm_below_brain:float=10.):
        
        ### nnUNet setup
        raw_img = f'{self.nnunet_raw_path}/{self.raw_name}/imagesTr'
        os.makedirs(raw_img, exist_ok=True)        
        raw_lab = f'{self.nnunet_raw_path}/{self.raw_name}/labelsTr' 
        os.makedirs(raw_lab, exist_ok=True)        

        case_identifier = os.path.basename(patient_path).replace("-", "_")
        ### Preprocessing the data
        ct, pet, mask = preprocess_case(patient_path)
        ct_cropped_brain, pet_cropped_brain, mask_cropped_brain = crop_below_brain(
            ct=ct, pet=pet, mask=mask, margin_mm=margin_mm_side_brain, case_path=patient_path, save=False)
        
        ct_cropped, pet_cropped, mask_cropped = crop_z_axis(
            ct_cropped_brain, pet_cropped_brain, mask_cropped_brain, margin_mm=margin_mm_below_brain, 
            case_path=patient_path, save=False)

        ## Checking that the mask contour has not been altered
        mask_npy = sitk.GetArrayFromImage(mask)
        mask_cropped_npy = sitk.GetArrayFromImage(mask_cropped)
        assert np.sum(mask_npy) == np.sum(mask_cropped_npy), \
            f"Mask was not retained after cropping. Original mask sum: {np.sum(mask_npy)}, cropped mask sum: {np.sum(mask_cropped_npy)}"
        

        ## Should we clip??
        ct_processed = ct_cropped
        pet_processed = pet_cropped
        mask_processed = mask_cropped

        sitk.WriteImage(
            ct_processed,
            f'{raw_img}/case_{case_identifier}_0001{self.extension}', 
            True
        )
        sitk.WriteImage(
            pet_processed,
            f'{raw_img}/case_{case_identifier}_0002{self.extension}', 
            True
        )
        sitk.WriteImage(
            mask_processed,
            f'{raw_lab}/case_{case_identifier}{self.extension}',
            True
        )

    def preprocess_data(self, multi_process:bool=True):
        patient_paths = [x for x in glob.glob(os.path.join(self.data_path, "*")) if os.path.isdir(x)]
        print(len(patient_paths), "cases found in the dataset")
        self.number_of_cases = len(patient_paths)
        if multi_process:
            with tqdm(total=len(patient_paths)) as pbar:
                with Pool(os.cpu_count()//3) as p:
                    for _ in p.imap_unordered(self.preprocess_data_single_case, patient_paths):
                        pbar.update()
        else:
            for patient_path in tqdm(patient_paths):
                self.preprocess_data_single_case(patient_path)

    
    def create_nnunetraw(self):
        self.preprocess_data()
        self.create_dataset_json()
        print(f"Dataset {self.raw_name} created in {self.nnunet_raw_path}")



def main(
    data_path: str,
    raw_name: str,
    nnunet_std: bool
):
    raw_gen = RawGenerator(data_path, raw_name, nnunet_std)
    raw_gen.create_nnunetraw()

if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Preprocess Hecktor2025 challenge dataset Task 1")
        parser.add_argument("-data_path", "-dp", type=str, help="Path to Hecktor2025 challenge dataset Task 1", default="/Data/Seb/Hecktor2025/Task_1")
        parser.add_argument("-raw_name", "-rn", type=str, help="Name of the raw dataset, should be DatasetXXX_zzzz", default="Dataset001_baseline")
        parser.add_argument("-nnunet_std", action="store_true", help="Use nnUNet standardization (default: False)")
        return parser.parse_args()
    args = parse_arguments()
    main(
        data_path=args.data_path,
        raw_name=args.raw_name,
        nnunet_std=args.nnunet_std
    )
    # export nnUNet_raw=/Data/Seb/Hecktor2025/nnUNet_raw



