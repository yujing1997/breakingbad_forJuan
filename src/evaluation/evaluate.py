import os
import json
import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from constants.labels import ids
from preprocessing.utils import preprocess_case
from preprocessing.cropping import crop_z_axis, crop_below_brain
from utils import compute_dice_score, compute_hauss_95

import numpy as np
import torch
import SimpleITK as sitk



def describe_array(array, percentile:bool=False):
    """
    Describe the array.
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if array.size == 0:
        return "No values"
    mean = np.mean(array)
    std = np.std(array)
    min_val = np.min(array)
    max_val = np.max(array)
    if percentile:
        percentile_90 = np.percentile(array, 90)
        percentile_95 = np.percentile(array, 95)
        percentile_99 = np.percentile(array, 99)
        median = np.median(array)
        return "{mean:.2f} +/- {std:.2f} [{min_val:.2f}, {max_val:.2f}] median {med:.2f}, percentiles: 90\% {percentile_90:.2f} 95\% {percentile_95:.2f} 99\% {percentile_99:.2f}".format(
            mean=mean, std=std, min_val=min_val, max_val=max_val, med=median, percentile_90=percentile_90, percentile_95=percentile_95, percentile_99=percentile_99
        )
    else:
        return "{mean:.2f} +/- {std:.2f} [{min_val:.2f}, {max_val:.2f}]".format(
            mean=mean, std=std, min_val=min_val, max_val=max_val
        )

class Evaluator:
    def __init__(self, ground_truth_folder:str, pred_folder:str, 
                 margin_mm_side_brain:float = 50.,margin_mm_below_brain:float = 10., 
                 out_dir:str = None):
        self.ground_truth_folder = ground_truth_folder
        self.pred_folder = pred_folder
        self.margin_mm_side_brain = margin_mm_side_brain
        self.margin_mm_below_brain = margin_mm_below_brain
        self.out_dir = out_dir if out_dir is not None else os.path.join(pred_folder, "evaluation")

    def evaluate_dataset(self):
        hd_scores = {}
        dice_scores = {}
        patient_ids = []
        for f in os.listdir(self.pred_folder):
            if f.endswith('.nii.gz'):
                patient_ids.append(f.split('.')[0].replace('case_', '').replace('_', '-'))
        for patient_id in tqdm.tqdm(patient_ids, total=len(patient_ids), desc="Evaluating predictions..."):
            print(f"Evaluating patient {patient_id}")
            hd, dice = self.evaluate_one_patient(patient_id)
            hd_scores[patient_id] = hd
            dice_scores[patient_id] = dice
        # Saving evaluation metrics into a json file
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        with open(os.path.join(self.out_dir, "evaluation_metrics.json"), "w") as f:
            json.dump({
                "hd_scores": hd_scores,
                "dice_scores": dice_scores
            }, f, indent=4)
        


    def evaluate_one_patient(self, patient_id:str):


        pred_path = os.path.join(self.pred_folder, f"case_{patient_id.replace('-', '_')}.nii.gz")
    
        print("Preprocessing mask...")
        patient_path = os.path.join(self.ground_truth_folder, patient_id)
        ct, pet, mask = preprocess_case(patient_path)
        ct_cropped_brain, pet_cropped_brain, mask_cropped_brain = crop_below_brain(
            ct=ct, pet=pet, mask=mask, margin_mm=self.margin_mm_side_brain, case_path=patient_path, save=False)
        
        ct_cropped, pet_cropped, mask_cropped = crop_z_axis(
            ct_cropped_brain, pet_cropped_brain, mask_cropped_brain, margin_mm=self.margin_mm_below_brain, 
            case_path=patient_path, save=False)
        gt_npy = sitk.GetArrayFromImage(mask_cropped)
        gt_tsr = torch.from_numpy(gt_npy)

        pred = sitk.ReadImage(pred_path)
        pred_npy = sitk.GetArrayFromImage(pred)
        pred_tsr = torch.from_numpy(pred_npy)
        print("Computing HD...")
        hd = compute_hauss_95(gt_tsr, pred_tsr)
        print("Computing DS...")
        dice = compute_dice_score(pred_tsr, gt_tsr)
        print("============================================")
        print("HD95:", hd)
        print("============================================")
        print("Dice:", dice)
        return hd, dice

    def inspect_metrics(self, results_path:str, k:int=3):
        names = {v: k for k, v in ids.items() if k != "background"}

        with open(results_path, "r") as f:
            metrics = json.load(f)
        hd_scores = metrics["hd_scores"]
        dice_scores = metrics["dice_scores"]
        for organ in names.values():
            print(f"=========================Organ: {organ}")
            oar_hd_scores = {k: v[organ] for k, v in hd_scores.items() if (organ in v) and not np.isnan(v[organ])}
            oar_dice_scores = {k: v[organ] for k, v in dice_scores.items() if organ in v}
            print("=========================HD95 Scores:")
            hd_scores_array = np.array(list(oar_hd_scores.values()))
            hd_patients = list(oar_hd_scores.keys())
            print(describe_array(hd_scores_array, percentile=True))
            # Identifying k worst patients
            sorted_indexes_hd = np.argsort(hd_scores_array)[::-1]
            print("Worst patients (HD95):")
            for i in range(k):
                idx = sorted_indexes_hd[i]
                print(f"{hd_patients[idx]}: {hd_scores_array[idx]:.2f}")
            print("=========================Dice Scores:")
            ds_scores_array = np.array(list(oar_dice_scores.values()))
            ds_patients = list(oar_dice_scores.keys())
            print(describe_array(ds_scores_array, percentile=True))
            # Identifying k worst patients
            sorted_indexes_ds = np.argsort(ds_scores_array)
            print("Worst patients (Dice):")
            for i in range(k):
                idx = sorted_indexes_ds[i]
                print(f"{ds_patients[idx]}: {ds_scores_array[idx]:.2f}")




if __name__ == "__main__":
    exp = "Dataset002_pet_ct_noMD" # "Dataset004_pet_ct_MD" # "Dataset001_pet_zscore_noMD" # "Dataset002_pet_ct_noMD"
    data_path = "/Data/Seb/Hecktor2025/Task_1" #"/home/sebquet/HECKTOR2025/Data/Task 1/""
    nnunet_results = os.environ.get("nnUNet_results")
    nnunet_result_path = f"{nnunet_results}/{exp}/"
    checkpoint = "best"
    evaluator = Evaluator(
        ground_truth_folder=data_path,
        pred_folder = os.path.join(nnunet_result_path, f"nnUNetTrainer__nnUNetPlans__3d_fullres_bs8/fold_0/validation_{checkpoint}/")
    )

    # evaluator.evaluate_one_patient(patient_id="CHUM-011")
    # evaluator.evaluate_dataset() 
    evaluator.inspect_metrics(
        os.path.join(nnunet_result_path,
                    f"nnUNetTrainer__nnUNetPlans__3d_fullres_bs8/fold_0/validation_{checkpoint}/evaluation/evaluation_metrics_{checkpoint}.json")   
    )