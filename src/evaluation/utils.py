import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from constants.labels import ids
names = {v: k for k, v in ids.items()}

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import SimpleITK as sitk
import torch.nn.functional as F

def compute_dice_score(prediction: torch.Tensor, target: torch.Tensor, num_classes:int=31):
    """
    Expects both tensors as [D, H, W]
    """
    if torch.cuda.is_available():
        prediction = prediction.cuda()
        target = target.cuda()
    # make [B, 1, H, W, D]
    prediction = prediction.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
    target = target.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
    # make one hot
    metric = DiceMetric(include_background=False, reduction="none", num_classes=num_classes)
    result = metric(prediction, target)[0].cpu()
    return_data = {}
    for channel in torch.unique(target).tolist()[1:]:
        return_data[names[channel]] = result[channel-1].item()
    return return_data


def compute_hauss_95(prediction: torch.Tensor, target: torch.Tensor, all_classes_at_once:bool=False):
    """
    Expects both tensors as [D, H, W]
    """
    if all_classes_at_once:
        # This uses SO MUCH memeory
        
        # one hot [D, H, W, C]
        target = F.one_hot(target.long(), num_classes=-1).int()
        prediction = F.one_hot(prediction.long(), num_classes=target.shape[1]).int()
        # if torch.cuda.is_available():
        #     prediction = prediction.cuda()
        #     target = target.cuda()
        # make [1, C, H, W, D]
        prediction = prediction.permute(3, 1, 2, 0).unsqueeze(0)
        target = target.permute(3, 1, 2, 0).unsqueeze(0)
        # make one hot
        metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
        result = metric(prediction, target)[0].cpu()
        print(result)
        return_data = {}
        for i in torch.unique(target).tolist()[1:]:
            return_data[names[i]] = result[i-1].item()
        return return_data
    else:
        print("Compute HD95 Channel by channel to save memory")
        return_data = {}
        for channel in torch.unique(target).tolist()[1:]:
            # print("Compute HD95 for channel ", channel)
            temp_target = (target == channel).unsqueeze(-1).int()
            temp_prediction = (prediction == channel).unsqueeze(-1).int()
            
            temp_target = temp_target.permute(3, 1, 2, 0).unsqueeze(0)
            temp_prediction = temp_prediction.permute(3, 1, 2, 0).unsqueeze(0)
            metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
            result = metric(temp_prediction, temp_target)[0].cpu()
            return_data[names[channel]] = result.item()
        return return_data
        


if __name__ == "__main__":
    import os
    from preprocessing.utils import preprocess_case
    from preprocessing.cropping import crop_z_axis, crop_below_brain
    margin_mm_side_brain = 50.
    margin_mm_below_brain = 10.

    ground_truth_path = "/home/sebquet/HECKTOR2025/Data/Task 1/CHUM-011/CHUM-011.nii.gz"
    pred_path = "/home/sebquet/HECKTOR2025/nnUNet_results/Dataset002_pet_ct_noMD/nnUNetTrainer__nnUNetPlans__3d_fullres_bs8/fold_0/validation/case_CHUM_011.nii.gz"
 
    print("Preprocessing mask...")
    patient_path = os.path.dirname(ground_truth_path)
    ct, pet, mask = preprocess_case(patient_path)
    ct_cropped_brain, pet_cropped_brain, mask_cropped_brain = crop_below_brain(
        ct=ct, pet=pet, mask=mask, margin_mm=margin_mm_side_brain, case_path=patient_path, save=False)
    
    ct_cropped, pet_cropped, mask_cropped = crop_z_axis(
        ct_cropped_brain, pet_cropped_brain, mask_cropped_brain, margin_mm=margin_mm_below_brain, 
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
