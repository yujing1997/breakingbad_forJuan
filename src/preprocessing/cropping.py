import os 
from pathlib import Path
import json

import SimpleITK as sitk
import numpy as np
from typing import Tuple, List
from scipy import ndimage

########################## GENERAL USE ##########################

def sitk_crop(image, bounding_box):
    """
    Crop the image to the bounding box
    """
    return sitk.RegionOfInterest(
        image,
        bounding_box[int(len(bounding_box) / 2) :],
        bounding_box[0 : int(len(bounding_box) / 2)],
    )


def crop_volumes_to_volume_fov(
    volume: sitk.Image, *other_volumes: sitk.Image
) -> Tuple[sitk.Image]:
    volume_FOV_bounds = get_air_bounds(volume)
    return (
        sitk_crop(volume, volume_FOV_bounds),
        *[sitk_crop(x, volume_FOV_bounds) for x in other_volumes],
    )


########################## AIR + KNOWLEDGE BASED CROPPING ##########################


def locate_brain_on_pet(pet:sitk.Image, suv_threshold:float=4., threshold_comp_size:float=100., 
                        case_id:str= "case_xxx", verbose:bool=False) -> sitk.Image:
    """
    Standardized Uptake Values threshold allows to identify regions of the body 
    with large quantity of radiotracer. We assume the biggest region is the brain. 
    We separate the mask identified by the threshold in connected components and 
    keep the bigger one. 
    """
    # print("threshold_comp_size should be realted to the size of the smallest lesion as well"
    #       "in case the brain is not part of the PET scan....")
    # exit()
    npy_array = sitk.GetArrayFromImage(pet)  # Ensure the volume is loaded correctly
    masked_pet = np.where(npy_array > suv_threshold, 1, 0)  # Masking the volume to remove non-air values
    assert np.any(masked_pet == 1) , \
        f"No pet SUV is greater than {suv_threshold}"
    # Identify the connected components
    s = np.ones((3, 3, 3), dtype=int)
    labeled_mask, num_labels = ndimage.label(masked_pet, structure=s)
    # Compute the size of each component
    component_sizes_voxel = np.bincount(labeled_mask.ravel()) 
    component_sizes = component_sizes_voxel * np.prod(pet.GetSpacing()) / 1000
    if verbose:
        print("component_sizes_voxel", component_sizes_voxel, "for case", case_id)
        print("COmponent sizes in cm3:", component_sizes, "for case", case_id)
    # Create a mask of component labels that meet the size threshold
    # Note: component_sizes[0] is the background (label 0), so ignore it
    
    potential_brains_comp_idx = np.where(component_sizes >= threshold_comp_size)[0].tolist()
    potential_brains_comp_idx.remove(0)  # Remove the background component (label 0)
    if verbose:
        print("Potential brain components indices:", potential_brains_comp_idx, "for case", case_id)
    # Identify the connected components
    # # keep the biggest component
    # potential_brains_comp_idx = []
    # for i in range(1, num_labels + 1):
    #     # Size of component in cm3
    #     component_size = np.sum(labeled_mask == i) * np.prod(pet.GetSpacing()) / 1000
    #     print(f"Component {i} size: {component_size} for {case_id}")
    #     if component_size > threshold_comp_size:
    #         potential_brains_comp_idx.append(i)
        
    # The brain component is the one that is the highest in the body
    # There might also be the bladder. 
    brain_component = None
    max_z_idx = 0
    for i in potential_brains_comp_idx:
        z_indexes = np.where(labeled_mask==i)[0]
        if verbose:
            print("np.max(z_indexes)", np.max(z_indexes))
        if np.max(z_indexes) > max_z_idx:
            brain_component = i
            max_z_idx = np.max(z_indexes)

    masked_pet = np.where(labeled_mask == brain_component, 1, 0)  # Keep only the brain component

    masked_pet_sitk = sitk.GetImageFromArray(masked_pet)
    masked_pet_sitk.CopyInformation(pet)  # Copy the information from the original volume
    # Change dtype to uint8 for binary mask
    masked_pet_sitk = sitk.Cast(masked_pet_sitk, sitk.sitkUInt8)

    return masked_pet_sitk

def get_air_bounds(volume):
    inside_value = 0
    outside_value = 1
    bin_image = sitk.OtsuThreshold(volume, inside_value, outside_value)

    # Get the bounding box of the anatomy
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(bin_image)
    bounding_box_air = label_shape_filter.GetBoundingBox(outside_value)
    return bounding_box_air


def get_boundaries_below_brain(brain:sitk.Image, margin_mm:float=0.0, case_id:str="case_xxx") -> Tuple[float, float, float, float]:

    print("brain.GetSpacing()", brain.GetSpacing())
    margin_x = int(margin_mm / brain.GetSpacing()[0])  # Convert margin to pixels
    margin_y = int(margin_mm / brain.GetSpacing()[1])  # Convert margin to pixels
    margin_z = 0
    # numpy is in z, y, x against x, y, z for sitk
    brain_npy = sitk.GetArrayFromImage(brain)
    print(f"Brain shape: {brain_npy.shape}, Brain dtype: {brain_npy.dtype} for case {case_id}")
    print("margin z ", margin_z, "unused for cropping below or above the brain")
    print("margin_y ", margin_y)
    print("margin_x ", margin_x)
    brain_coords = np.where(brain_npy == 1)
    min_z = int(np.min(brain_coords[0]))  # z coordinate: lowest point of the brain
    max_z = int(np.max(brain_coords[0]))  # z coordinate: highest point of the brain
    min_y = int(np.min(brain_coords[1]))  # y coordinate: nose 
    max_y = int(np.max(brain_coords[1]))  # y coordinate: back of the head 
    min_x = int(np.min(brain_coords[2]))  # x coordinate: left ear
    max_x = int(np.max(brain_coords[2]))  # x coordinate: right ear
    print(f"Brain coordinates for case {case_id}: min_z={min_z}, max_z={max_z}, min_y={min_y}, max_y={max_y}, min_x={min_x}, max_x={max_x}")

    ## The bounding box is [origin_x, origin_y, origin_z, size_x, size_y, size_z]
    origin_x = int(max(0, min_x - margin_x))
    origin_y = int(max(0, min_y - margin_y))
    origin_z = 0
    bb = [
        origin_x,
        origin_y,
        0, # we only crop z above the head
        int(min(max_x - min_x + 2 * margin_x, brain_npy.shape[2] - origin_x)),  # x size
        int(min(max_y - min_y + 2 * margin_y, brain_npy.shape[1] - origin_y)),  # y size
        # Cropping just above brain
        int(min(max_z - 0 + 2 * margin_z, brain_npy.shape[0] - origin_z))  # z size
    ]
    print(f"Bounding box below brain for case {case_id}: {bb}")
    return bb



def crop_air_v2(ref_volume:sitk.Image, volumes_to_crop:List[sitk.Image]):
    bounding_box_air = get_air_bounds(ref_volume)
    ref_volume_cropped_air = sitk_crop(ref_volume, bounding_box_air)
    cropped_volumes = []
    if len(volumes_to_crop) > 0:   
        for mask in volumes_to_crop: 
            cropped_volumes.append(sitk_crop(mask, bounding_box_air))
    return ref_volume_cropped_air, *tuple(cropped_volumes)


def crop_one_axis_v2(
    sitk_volume,
    axis=2,
    first_contour_mm=0,
    last_contour_mm=900,
    margin_mm=20,
    bounding_box=None,
):
    """_summary_

    Args:
        axis (int, optional): x, y or z axis (0,1 or 2). Defaults to 2.
        size_mm (int, optional): size computed in the dataset. Defaults to 450.
        margin_mm (int, optional): margin to add to the size. Defaults to 5.
        end_axis (bool, optional): if True, crop the end of the volume, oif false, crop the begining. Defaults to False.
    """

    # Get the spacing and size of the volume
    axis_spacing = sitk_volume.GetSpacing()[axis]
    slice_nb = sitk_volume.GetSize()[axis]
    axis_size_mm = np.abs(axis_spacing * slice_nb)
    print(f"Axis {axis} size: {axis_size_mm}")
    # Taking absolute values for distances
    first_contour_mm = abs(first_contour_mm)
    last_contour_mm = abs(last_contour_mm)

    # Getting a margin of safety for the cropping
    last_contour_mm = min(last_contour_mm + margin_mm, axis_size_mm)
    first_contour_mm = max(first_contour_mm - margin_mm, 0)

    print(f"First contour: {first_contour_mm}, Last contour: {last_contour_mm}")
    # bounding box are given as [x_start, y_start, z_start, x_size, y_size, z_size].
    if bounding_box is None:
        bounding_box = [
            0,
            0,
            0,
            sitk_volume.GetSize()[0],
            sitk_volume.GetSize()[1],
            sitk_volume.GetSize()[2],
        ]

    # Lower part of the axis
    start_crop = int(np.floor(axis_size_mm - last_contour_mm) / axis_spacing)

    # start_crop = int(np.floor(first_contour_mm / axis_spacing))
    bounding_box[axis] = start_crop
    bounding_box[3 + axis] = slice_nb - start_crop

    # Upper part of the axis
    diff_pixel = int(np.ceil(first_contour_mm / axis_spacing))
    bounding_box[3 + axis] -= diff_pixel

    return bounding_box


def get_mask_maxmm_from_top_brain(mask, case:str="case_xxx") -> float:
    spacings = mask.GetSpacing()
    size = mask.GetSize()
    print(f"Mask size: {size}, Spacings: {spacings}")
    origin_x, origin_y, origin_z = mask.GetOrigin()
    direction = mask.GetDirection()
    assert np.all(np.array(direction) == np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))), (
        "Direction should be identity matrix for this function to work properly."
    )
    mask_npy = sitk.GetArrayFromImage(mask)
    print(f"Mask shape: {mask_npy.shape}, Mask dtype: {mask_npy.dtype}")
    non_zero_coord = np.where(mask_npy != 0)
    if len(non_zero_coord[0]) == 0:
        print(f"Warning: No non-zero coordinates found in mask for case {case}. Returning zeros.")
        return 0.
    print("Max_z from top (min z coord) =", np.min(non_zero_coord[0]), "for case", case)
    max_z_mm_from_top_brain = np.abs(size[2] - np.min(non_zero_coord[0])) * spacings[2]
    return max_z_mm_from_top_brain

def crop_z_axis(ct:sitk.Image, pet:sitk.Image, mask:sitk.Image=None, margin_mm:float=50.0, 
                save:bool=False, case_path:str=None) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
    """
    This function crops only the z-axis of the CT, PET and mask images, and only remove parts below the brain.
    WARNING: This function assumes that the image are already cropped to the brain using crop_below_brain() 
    function.

    Args:
        ct (sitk.Image): _description_
        pet (sitk.Image): _description_
        mask (sitk.Image): _description_
        margin_mm (float, optional): _description_. Defaults to 50.0.

    Returns:
        Tuple[sitk.Image, sitk.Image, sitk.Image]: _description_
    """
    ct_shape = ct.GetSize()
    boundries_path = os.path.join(
        str(Path(__file__).parents[1]), "constants", 
        "biggest_boundaries_masks_cropped_below_brain_margin_50.0.json")
    if not os.path.exists(boundries_path):
        raise FileNotFoundError(f"Boundaries file {boundries_path} does not exist. Please run the script to generate it.")
    with open(boundries_path, "r") as f:
        largest_boundaries = json.load(f)
    distance_max_below_tipofbrain_mm = largest_boundaries["max_z_from_top_brain_mm"]
    distance_max_below_tipofbrain_mm += margin_mm  # Add margin to the distance below the tip of the brain
    distance_max_below_tipofbrain_voxel = int(np.ceil(
        distance_max_below_tipofbrain_mm / ct.GetSpacing()[2]
    ))  # Convert to voxel space
    origin_bb_z = int(max(0, ct_shape[2] - distance_max_below_tipofbrain_voxel))
    
    bb = [
        0,
        0,
        # Cropping the z-axis until we reach distance_max_below_tipofbrain
        origin_bb_z,
        ct_shape[0],
        ct_shape[1],
        ct_shape[2] - origin_bb_z

    ]
    print("bounding box for cropping z-axis:", bb)
    print("CT shape before cropping z-axis:", ct_shape)
    ct_cropped = sitk_crop(ct, bb)
    pet_cropped = sitk_crop(pet, bb)
    if not (mask is None):
        mask_cropped = sitk_crop(mask, bb)
    if save:
        assert not (case_path is None), "case_path must be provided if save is True"
        sitk.WriteImage(
            ct_cropped, 
            os.path.join(case_path, f"{os.path.basename(case_path)}__CT_cropped_z_axis.nii.gz"), 
            True
        )
        sitk.WriteImage(
            pet_cropped, 
            os.path.join(case_path, f"{os.path.basename(case_path)}__PT_cropped_z_axis.nii.gz"), 
            True
        )
        if not (mask is None):
            sitk.WriteImage(
                mask_cropped, 
                os.path.join(case_path, f"{os.path.basename(case_path)}__mask_cropped_z_axis.nii.gz"), 
                True
            )
    if mask is None:
        return ct_cropped, pet_cropped
    else:
        return ct_cropped, pet_cropped, mask_cropped


def crop_below_brain(ct:sitk.Image, pet:sitk.Image, mask:sitk.Image=None, margin_mm:float=50.0,
                     suv_threshold_percentile:float=95., 
                     case_path:str="./data/case_xxx", save:bool=False):
    """
    Crops the pet, ct and mask below the brain. Above brain is cropped out. ON the side 
    of the brain we keep a margin of margin_mm.
    """
    case_id = os.path.basename(case_path)

    pet_npy = sitk.GetArrayFromImage(pet)
    ct_npy = sitk.GetArrayFromImage(ct)
    ct_body = np.where((ct_npy>-500) & (ct_npy<1000), True, False)
    pet_body = pet_npy[ct_body]
    ct_body_sitk = sitk.GetImageFromArray(ct_body.astype(np.uint8))
    ct_body_sitk.CopyInformation(ct)
    suv_threshold = np.round(np.percentile(pet_body, suv_threshold_percentile), 2)
    brain = locate_brain_on_pet(pet, suv_threshold, case_id=case_id)  # This is just for debugging, remove later
    bounding_below_brain = get_boundaries_below_brain(brain, margin_mm)
    ct_cropped_brain = sitk_crop(ct, bounding_below_brain)
    pet_cropped_brain = sitk_crop(pet, bounding_below_brain)
    if not (mask is None):
        mask_cropped_brain = sitk_crop(mask, bounding_below_brain)

    if save:
        sitk.WriteImage(
            ct_body_sitk,
            os.path.join(case_path, f"{os.path.basename(case_path)}__CT_body_mask.nii.gz"),
            True
        )
        sitk.WriteImage(
            brain, 
            os.path.join(case_path, f"{os.path.basename(case_path)}__brain_mask_thresolded{suv_threshold}suv.nii.gz"),
            True
        )
        sitk.WriteImage(
            ct_cropped_brain, 
            os.path.join(case_path, f"{os.path.basename(case_path)}__CT_cropped_below_brain.nii.gz"), 
            True)  
        sitk.WriteImage(
            pet_cropped_brain, 
            os.path.join(case_path, f"{os.path.basename(case_path)}__PT_cropped_below_brain.nii.gz"), 
            True)
        if not (mask is None):
            sitk.WriteImage(
                mask_cropped_brain, 
                os.path.join(case_path, f"{os.path.basename(case_path)}__mask_cropped_below_brain.nii.gz"), 
                True)  
    if mask is None:
        return ct_cropped_brain, pet_cropped_brain
    else:
        return ct_cropped_brain, pet_cropped_brain, mask_cropped_brain


def crop_air(ct:sitk.Image, pet:sitk.Image, mask:sitk.Image, 
             cropair_pet:bool=True, clip_ct:bool=False) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:

    # To decide which volume is used for cropping
    if cropair_pet:
        pet_cropped_air, ct_cropped_air, mask_cropped_air = crop_air_v2(pet, [ct, mask])
    else:
        if clip_ct:
            ct_used_for_cropping = sitk.Clamp(ct, upperBound=500, lowerBound=-500)
        else:
            ct_used_for_cropping = ct
        ct_cropped_air, pet_cropped_air, mask_cropped_air = crop_air_v2(ct_used_for_cropping, [pet, mask])

    return ct_cropped_air, pet_cropped_air, mask_cropped_air

# getting lowest mask and highest
def get_mask_extremities(mask, case):
    spacings = mask.GetSpacing()
    size = mask.GetSize()
    print(f"Mask size: {size}, Spacings: {spacings}")
    origin_x, origin_y, origin_z = mask.GetOrigin()
    direction = mask.GetDirection()
    assert np.all(np.array(direction) == np.array((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))), (
        "Direction should be identity matrix for this function to work properly."
    )
    mask_npy = sitk.GetArrayFromImage(mask)
    print(f"Mask shape: {mask_npy.shape}, Mask dtype: {mask_npy.dtype}")
    non_zero_coord = np.where(mask_npy != 0)
    if len(non_zero_coord[0]) == 0:
        print(f"Warning: No non-zero coordinates found in mask for case {case}. Returning zeros.")
        return (size[2]*spacings[2], 
                0, 
                size[1]*spacings[1],
                0,
                size[0]*spacings[0],
                0,
                origin_x, origin_y, origin_z)
    last_z_mm = np.max(non_zero_coord[0]) * spacings[2]
    first_z_mm = np.min(non_zero_coord[0]) * spacings[2]
    print(f"Non-zero coordinates: from {np.min(non_zero_coord[0])} to {np.max(non_zero_coord[0])} in z axis")
    last_y_mm = np.max(non_zero_coord[1]) * spacings[1]
    first_y_mm = np.min(non_zero_coord[1]) * spacings[1]
    print(f"Non-zero coordinates: from {np.min(non_zero_coord[1])} to {np.max(non_zero_coord[1])} in y axis")
    last_x_mm = np.max(non_zero_coord[2]) * spacings[0]
    first_x_mm = np.min(non_zero_coord[2]) * spacings[0]
    print(f"Non-zero coordinates: from {np.min(non_zero_coord[2])} to {np.max(non_zero_coord[2])} in x axis")

    return first_x_mm, last_x_mm, first_y_mm, last_y_mm, first_z_mm, last_z_mm, origin_x, origin_y, origin_z
