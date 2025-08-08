import os
import glob
import SimpleITK as sitk
import numpy as np
from typing import List, Tuple


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

def get_image_metadata(image):
    print("image size ", image.GetSize())
    print("img origin :", image.GetOrigin())
    print("img spacing :", image.GetSpacing())
    print("img direction :", image.GetDirection())
    print("img nb components per pixel :", image.GetNumberOfComponentsPerPixel())

    print("img width :", image.GetWidth())
    print("img heigth :", image.GetHeight())
    print("img depth :", image.GetDepth())
    print("img dimension :", image.GetDimension())
    print("img GetPixelIDValue :", image.GetPixelIDValue())
    print("img GetPixelIDTypeAsString :", image.GetPixelIDTypeAsString())


def get_mask(mask_path:str) -> sitk.Image:
    mask = sitk.ReadImage(mask_path)
    return mask

def resample_mask_to_ct(mask:sitk.Image, ct:sitk.Image) -> sitk.Image:
    """
    Resample mask to CT image space.
    This function assumes that the mask image is in the same space as the CT image.
    Args:
        mask (sitk.Image): Contour image.
        ct (sitk.Image): CT image.
    Returns:
        sitk.Image: Resampled mask image.
    """
    new_mask  = sitk.Resample(mask, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
    return new_mask

def preprocess_case(case_path:str) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
    patient_code = os.path.basename(case_path)
    mask_potential_paths = glob.glob(os.path.join(case_path, f"{patient_code}.nii.gz"))
    assert len(mask_potential_paths) == 1, f"Expected one mask for {case_path}, found {len(mask_potential_paths)}"
    ct_potential_paths = glob.glob(os.path.join(case_path, f"{patient_code}__CT.nii.gz"))
    assert len(ct_potential_paths) == 1, f"Expected one CT for {case_path}, found {len(ct_potential_paths)}"
    pet_potential_paths = glob.glob(os.path.join(case_path, f"{patient_code}__PT.nii.gz"))
    assert len(pet_potential_paths) == 1, f"Expected one PET for {case_path}, found {len(pet_potential_paths)}"
    mask_path = mask_potential_paths[0]
    ct_path = ct_potential_paths[0]
    pet_path = pet_potential_paths[0]
    ct = sitk.ReadImage(ct_path)
    mask = resample_mask_to_ct(get_mask(mask_path), ct)
    assert ct.GetSize() == mask.GetSize(), f"CT and mask sizes do not match for {case_path}: CT size {ct.GetSize()}, mask size {mask.GetSize()}"
    pet = resample_pet_to_ct(sitk.ReadImage(pet_path), ct)
    return ct, pet, mask

def resample_pet_to_ct(pet:sitk.Image, ct:sitk.Image) -> sitk.Image:
    """
    Resample PET to CT image space.
    This function assumes that the PET image is in the same space as the CT image.
    Args:
        pet (sitk.Image): PET image.
        ct (sitk.Image): CT image.
    Returns:
        sitk.Image: Resampled PET image.
    """
    new_pet  = sitk.Resample(pet, ct, sitk.Transform(), sitk.sitkBSpline, 0, pet.GetPixelID())
    return new_pet

def clip_ct(sitk_ct:sitk.Image):
    ct_npy = sitk.GetArrayFromImage(sitk_ct)
    ct_npy[ct_npy < -1000] = -1000
    clipped_ct = sitk.GetImageFromArray(ct_npy)
    clipped_ct.CopyInformation(sitk_ct)
    return clipped_ct

def pad_mask(mask_to_pad, spacing, size, origin, direction):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    return resampler.Execute(mask_to_pad)