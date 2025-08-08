import os
import glob
import tqdm
import json
import inspect
from pathlib import Path
import concurrent.futures

import numpy as np
import SimpleITK as sitk
from cropping import get_mask_extremities, get_mask_maxmm_from_top_brain, crop_air, crop_below_brain
from utils import preprocess_case


if __name__ == "__main__":
    
    set_root = "/Data/Seb/Hecktor2025/Task_1/"
    assert os.path.exists(set_root), f"{set_root} not exist"
    cases = [x for x in glob.glob(f"{set_root}/*") if os.path.isdir(x)]
    MARGIN_MM = 50.0
    multiprocessing = True

    def process_case(case_path, cropair_pet:bool=True, clip_ct:bool=False, margin_mm:float=MARGIN_MM):
        print(f"Processing {case_path}")
        ct, pet, mask = preprocess_case(case_path)
        mask_sum_contours = np.sum(sitk.GetArrayFromImage(mask))
        ct_cropped_air, pet_cropped_air, mask_cropped_air = crop_below_brain(ct=ct, pet=pet, mask=mask, margin_mm=margin_mm, case_path=case_path, save=True)
        mask_sum_cropped_contours = np.sum(sitk.GetArrayFromImage(mask_cropped_air))
        assert mask_sum_cropped_contours == mask_sum_contours, \
        f"Mask sum contours before cropping {mask_sum_contours} does not match after cropping {mask_sum_cropped_contours} for patient {os.path.basename(case_path)}"

        # ct_cropped_air, pet_cropped_air, mask_cropped_air = crop_air(ct, pet, mask, cropair_pet, clip_ct)
        if cropair_pet:
            suffix = "cropped_air_from_pet"
        else:
            suffix = "cropped_air_from_ct"
        first_x_mm_cropped, last_x_mm_cropped, first_y_mm_cropped, last_y_mm_cropped, first_z_mm_cropped, last_z_mm_cropped, origin_x_cropped, origin_y_cropped, origin_z_cropped = get_mask_extremities(mask_cropped_air, case_path)
        max_z_mm_from_top_brain = get_mask_maxmm_from_top_brain(mask_cropped_air, case_path)
        return max_z_mm_from_top_brain, first_x_mm_cropped, last_x_mm_cropped, first_y_mm_cropped, last_y_mm_cropped, first_z_mm_cropped, last_z_mm_cropped, origin_x_cropped, origin_y_cropped, origin_z_cropped, os.path.basename(case_path)

    max_z_mms_from_top_brain = []
    first_x_mms_cropped, last_x_mms_cropped, first_y_mms_cropped, last_y_mms_cropped, first_z_mms_cropped, last_z_mms_cropped, origins_x_cropped, origins_y_cropped, origins_z_cropped, case_list_cropped = [], [], [], [], [], [], [], [], [], []

    if not multiprocessing:
        for case in tqdm.tqdm(cases, desc="Processing cases"):
            print("POrocessing case:", case)
            if not "CHUS-100" in case:
                continue
            max_z_mm_from_top_brain, first_x_mm, last_x_mm, first_y_mm, last_y_mm, first_z_mm, last_z_mm, origin_x, origin_y, origin_z, case_path = process_case(case)
            max_z_mms_from_top_brain.append(max_z_mm_from_top_brain)
            first_x_mms_cropped.append(first_x_mm)
            last_x_mms_cropped.append(last_x_mm)
            first_y_mms_cropped.append(first_y_mm)
            last_y_mms_cropped.append(last_y_mm)
            first_z_mms_cropped.append(first_z_mm)
            last_z_mms_cropped.append(last_z_mm)
            origins_x_cropped.append(origin_x)
            origins_y_cropped.append(origin_y)
            origins_z_cropped.append(origin_z)
            case_list_cropped.append(case_path)
        exit()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            results = tqdm.tqdm(executor.map(process_case, cases), total=len(cases))
            for result in results:
                max_z_mms_from_top_brain.append(result[0])
                first_x_mms_cropped.append(result[1])
                last_x_mms_cropped.append(result[2])
                first_y_mms_cropped.append(result[3])
                last_y_mms_cropped.append(result[4])
                first_z_mms_cropped.append(result[5])
                last_z_mms_cropped.append(result[6])
                origins_x_cropped.append(result[7])
                origins_y_cropped.append(result[8])
                origins_z_cropped.append(result[9])
                case_list_cropped.append(result[10])

    max_z_from_top_brain_index = np.argmax(max_z_mms_from_top_brain)
    max_z_from_top_brain_patient = case_list_cropped[max_z_from_top_brain_index]
    max_z_from_top_brain_mm = max_z_mms_from_top_brain[max_z_from_top_brain_index]
    min_x_index = np.argmin(first_x_mms_cropped)
    min_x_mm = first_x_mms_cropped[min_x_index]
    min_x_patient = case_list_cropped[min_x_index]
    max_x_index = np.argmax(last_x_mms_cropped)
    max_x_patient = case_list_cropped[max_x_index]
    max_x_mm = last_x_mms_cropped[max_x_index]
    min_y_index = np.argmin(first_y_mms_cropped)
    min_y_mm = first_y_mms_cropped[min_y_index]
    min_y_patient = case_list_cropped[min_y_index]
    max_y_index = np.argmax(last_y_mms_cropped)
    max_y_patient = case_list_cropped[max_y_index]
    max_y_mm = last_y_mms_cropped[max_y_index]
    min_z_index = np.argmin(first_z_mms_cropped)
    min_z_mm = first_z_mms_cropped[min_z_index]
    min_z_patient = case_list_cropped[min_z_index]
    max_z_index = np.argmax(last_z_mms_cropped)
    max_z_patient = case_list_cropped[max_z_index]
    max_z_mm = last_z_mms_cropped[max_z_index]

    # Saving boundaries as dict
    boundaries = {
        "min_x_mm": min_x_mm,
        "min_x_patient": min_x_patient,
        "max_x_mm": max_x_mm,
        "max_x_patient": max_x_patient,
        "min_y_mm": min_y_mm,
        "min_y_patient": min_y_patient,
        "max_y_mm": max_y_mm,
        "max_y_patient": max_y_patient,
        "min_z_mm": min_z_mm,
        "min_z_patient": min_z_patient,
        "max_z_mm": max_z_mm,
        "max_z_patient": max_z_patient,
        "max_z_from_top_brain_mm": max_z_from_top_brain_mm,
        "max_z_from_top_brain_patient": max_z_from_top_brain_patient,
    }
    local_file_path = os.path.abspath( inspect.getfile(inspect.currentframe()))
    repo_src_path = Path(local_file_path).parents[1]
    boundaries_path = os.path.join(repo_src_path, "constants", f"biggest_boundaries_masks_cropped_below_brain_margin_{MARGIN_MM}.json")
    with open(boundaries_path, "w") as f:
        json.dump(boundaries, f, indent=4)