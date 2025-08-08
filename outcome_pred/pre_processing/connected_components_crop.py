"""
Adapted from Sebbers
"""

########################## Connected components ##########################
import os
import collections
import numpy as np
from typing import List, Union, Sequence, Dict
from math import ceil
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
# 0 is ct 1 is pet, ak

def process_connected_components(prediction_path: str):
    """
    Process connected components from a prediction mask
    
    Args:
        prediction_path: Path to the prediction mask file
    """
    mask = sitk.GetArrayFromImage(sitk.ReadImage(prediction_path))
    labeled_mask, num_labels = ndimage.label(mask)
    
    components = []
    for connected_comp_idx in range(1, num_labels + 1):
        component_array = np.array(
            labeled_mask == connected_comp_idx, dtype=np.uint8
        )
        components.append(component_array)
        # Do your cropping there if needed
    
    return components, labeled_mask, num_labels


########################## Cropping ##########################

import SimpleITK as sitk
import numpy as np

def sitk_crop(image, bounding_box):
    """
    Crop the image to the bounding box
    """
    return sitk.RegionOfInterest(
    image,
    # bounding_box[0:3] is the x_min, y_min, z_min
    bounding_box[int(len(bounding_box) / 2) :],
    # bounding_box[3:6] is the x_size, y_size, z_size
    bounding_box[0 : int(len(bounding_box) / 2)],
    )

# use this for bigger bounding boxes
# find the max bounding box size for all patients and define that as the bounding box size 

def crop_around_mask(volume: sitk.Image, margin_mm: float = 0., use_sitk: bool = True):
    """
    Gets cropping boundaries to crop any volume around the specified mask.
    
    Args: 
        volume: SimpleITK image containing the mask
        margin_mm: Margin in millimeters to add around the mask
        use_sitk: Whether to use SimpleITK (faster) or numpy approach
        
    Returns:
        tuple: (cropped_volume, bounding_box)
    """
    spacings = volume.GetSpacing()
    
    if use_sitk:
        # Using SimpleITK - faster approach
        non_zero_mask = sitk.BinaryThreshold(volume, lowerThreshold=1)

        # Get bounding box from contour
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(non_zero_mask)
        bounding_box = list(label_shape_filter.GetBoundingBox(1))

        # Add margin
        for i in range(0, 3):
            bounding_box[i] = max(0, bounding_box[i] - int(margin_mm / spacings[i]))
        for i in range(3, 6):
            bounding_box[i] = min(
                # min between bounding box with margin size and remaining volume shape from bounding box
                volume.GetSize()[i-3] - bounding_box[i-3],
                bounding_box[i] + int(margin_mm / spacings[i-3]) * 2
            )
    else:
        # Using numpy - slower but more flexible approach
        mask_array = sitk.GetArrayFromImage(volume)
        # Get nonzero mask indices efficiently
        nz = np.where(mask_array > 0)  # Faster than np.nonzero(mask_array)
        
        if len(nz[0]) == 0:
            raise ValueError("No nonzero voxels found in the mask.")
            
        # Compute min/max for each axis directly
        zmin, ymin, xmin = [int(np.min(a)) for a in nz]
        zmax, ymax, xmax = [int(np.max(a)) + 1 for a in nz]
        
        # Add margin and clamp to array shape
        shape = mask_array.shape
        zmin = max(0, zmin - int(margin_mm / spacings[2]))
        ymin = max(0, ymin - int(margin_mm / spacings[1]))
        xmin = max(0, xmin - int(margin_mm / spacings[0]))
        zmax = min(shape[0], zmax + int(margin_mm / spacings[2]))
        ymax = min(shape[1], ymax + int(margin_mm / spacings[1]))
        xmax = min(shape[2], xmax + int(margin_mm / spacings[0]))
        
        # bounding_box: [x, y, z, size_x, size_y, size_z]
        bounding_box = [
            xmin, ymin, zmin,
            xmax - xmin, ymax - ymin, zmax - zmin
        ]

    return sitk_crop(volume, bounding_box), bounding_box


####
def _issequence(obj):
    if isinstance(obj, (bytes, str)):
        return False
    return isinstance(obj, collections.abc.Sequence)

# Adapted from https://github.com/SimpleITK/SimpleITKUtilities/blob/b3d148cd8a0a354a279b84d3a5d3f4c7d09a8305/SimpleITK/utilities/resize.py#L77
def resample_volume(
    image: sitk.Image,
    new_spacing: Sequence[float] = [1.0, 1.0, 1.0],
    fill: bool = True,
    interpolator=sitk.sitkBSpline,
    fill_value: float = 0.0,
    use_nearest_extrapolator: bool = False,
    anti_aliasing_sigma: Union[None, float, Sequence[float]] = None,
) -> sitk.Image:
    """
    Resize an image to an arbitrary size while retaining the original image's spatial location.

    Allows for specification of the target image size in pixels, and whether the image pixels spacing should be
    isotropic. The physical extent of the image's data is retained in the new image, with the new image's spacing
    adjusted to achieve the desired size. The image is centered in the new image.

    Anti-aliasing is enabled by default.

    Runtime performance can be increased by disabling anti-aliasing ( anti_aliasing_sigma=0 ), and by setting
    the interpolator to sitkNearestNeighbor at the cost of decreasing image quality.

    :param image: A SimpleITK image.
    :param new_spacing: The new image spacing in mm.
    :param fill: If True, the output image will be new_size, and the original image will be centered in the new image
    with constant or nearest values used to fill in the new image. If False and isotropic is True, the output image's
    new size will be calculated to fit the original image's extent such that at least one dimension is equal to
    new_size.
    :param fill_value: Value used for padding.
    :param interpolator: Interpolator used for resampling.
    :param use_nearest_extrapolator: If True, use a nearest neighbor for extrapolation when resampling, overridding the
    constant fill value.
    :param anti_aliasing_sigma: If zero no antialiasing is performed. If a scalar, it is used as the sigma value in
     physical units for all axes. If None or a sequence, the sigma value for each axis is calculated as
     $sigma = (new_spacing - old_spacing) / 2$ in physical units. Gaussian smoothing is performed prior to resampling
     for antialiasing.
    :return: A SimpleITK image with desired size.
    """


    new_size = [
        int(round(osz * ospc / nspc)) 
        for osz, ospc, nspc in zip(image.GetSize(), image.GetSpacing(), new_spacing)
    ]

    if not fill:
        new_size = [
            ceil(osz * ospc / nspc)
            for ospc, osz, nspc in zip(image.GetSpacing(), image.GetSize(), new_spacing)
        ]

    center_cidx = [0.5 * (sz - 1) for sz in image.GetSize()]
    new_center_cidx = [0.5 * (sz - 1) for sz in new_size]

    new_origin_cidx = [0] * image.GetDimension()

    # The continuous index of the new center of the image, in the original image's continuous index space.
    for i in range(image.GetDimension()):
        new_origin_cidx[i] = center_cidx[i] - new_center_cidx[i] * (
            new_spacing[i] / image.GetSpacing()[i]
        )
    new_origin = image.TransformContinuousIndexToPhysicalPoint(new_origin_cidx)
    
    input_pixel_type = image.GetPixelID()

    if anti_aliasing_sigma is None:
        # (s-1)/2.0 is the standard deviation of the Gaussian kernel in index space, where s downsample factor defined
        # by nspc/ospc.
        anti_aliasing_sigma = [
            max((nspc - ospc) / 2.0, 0.0)
            for ospc, nspc in zip(image.GetSpacing(), new_spacing)
        ]
    elif not _issequence(anti_aliasing_sigma):
        anti_aliasing_sigma = [anti_aliasing_sigma] * image.GetDimension()

    if any([s < 0.0 for s in anti_aliasing_sigma]):
        raise ValueError("anti_aliasing_sigma must be positive, or None.")
    if len(anti_aliasing_sigma) != image.GetDimension():
        raise ValueError(
            "anti_aliasing_sigma must be a scalar or a sequence of length equal to the image dimension."
        )

    if all([s > 0.0 for s in anti_aliasing_sigma]):
        image = sitk.SmoothingRecursiveGaussian(image, anti_aliasing_sigma)
    else:
        for d, s in enumerate(anti_aliasing_sigma):
            if s > 0.0:
                image = sitk.RecursiveGaussian(image, sigma=s, direction=d)

    return sitk.Resample(
        image,
        size=new_size,
        outputOrigin=new_origin,
        outputSpacing=new_spacing,
        outputDirection=image.GetDirection(),
        defaultPixelValue=fill_value,
        interpolator=interpolator,
        useNearestNeighborExtrapolator=use_nearest_extrapolator,
        outputPixelType=input_pixel_type,
    )

def get_one_axis_bound(
        shape, size, max_fmio, min_fmio, name="X", verbose=True
    ):
        """
        Get bounding box for one axis, handling edge cases
        
        Args:
            shape: Original volume shape for this axis
            size: Desired size for this axis
            max_fmio: Maximum boundary from mask
            min_fmio: Minimum boundary from mask
            name: Axis name for logging
            verbose: Whether to print verbose information
            
        Returns:
            List with [min_bound, max_bound] for this axis
        """
        size_fmio = max_fmio - min_fmio
        if shape < size:
            if verbose:
                print(
                    "{} axis on volume is too small with size {} compared to {} we keep all voxels on that dimension and will pad later".format(
                        name, shape, size
                    )
                )
            bounds = [0, shape]
        else:
            if verbose:
                print(
                    "For {} we want size {} but we have size {} with fmio bounds and original volume size on {} axis is {} ".format(
                        name, size, size_fmio, name, shape
                    )
                )
            to_add = size - size_fmio
            if to_add <= 0:
                print("shape, size, max_fmio, min_fmio ", shape, size, max_fmio, min_fmio)
                updated_size = size + 16
                print(
                    """{} : FMIO bounds computed are already bigger than the proposed FMIO shape, we will find FMIO boundaries 
                    of a bigger size divisible by 16: {}.""".format(
                            name, 
                            updated_size
                        )
                    )
                return get_one_axis_bound(shape, updated_size, max_fmio, min_fmio, name, verbose)
            else:
                if max_fmio + int(np.ceil(to_add / 2)) > shape:
                    add_left = to_add - (shape - max_fmio)
                    if verbose:
                        print("{}max fmio : ".format(name), max_fmio)
                        print(
                            "What we would like bound to be: ",
                            max_fmio + int(np.ceil(to_add / 2)),
                        )
                        print(
                            "{} : Too big on the right we will go to the edge and remove {} pixels on the other side.".format(
                                name, add_left
                            )
                        )
                    bounds = [min_fmio - add_left, shape]
                elif min_fmio - int(np.floor(to_add / 2)) < 0:
                    add_right = to_add - (min_fmio)
                    if verbose:
                        print("{}min fmio : ".format(name), min_fmio)
                        print(
                            "What we would like bound to be: ",
                            min_fmio - int(np.floor(to_add / 2)),
                        )
                        print(
                            "{} : Too big on the left we will go to the edge and add {} pixels on the other side.".format(
                                name, add_right
                            )
                        )
                    bounds = [0, max_fmio + add_right]
                else:
                    if verbose:
                        print(
                            "{} : Centered. Adding {} pixels split on each side.".format(
                                name, to_add
                            )
                        )
                    bounds = [
                        min_fmio - int(np.floor(to_add / 2)),
                        max_fmio + int(np.ceil(to_add / 2)),
                    ]
        return bounds


def find_optimal_spacing_and_size(bounding_boxes_1mm: List[List], 
                                max_memory_gb: float = 10.0,
                                max_size_per_dim: int = 512) -> Dict:
    """
    Find optimal spacing and bounding box size for GPU memory constraints
    
    Args:
        bounding_boxes_1mm: List of bounding box sizes at 1mm spacing
        max_memory_gb: Maximum GPU memory in GB
        max_size_per_dim: Maximum size per dimension
        
    Returns:
        Dict with optimal parameters
    """
    # Convert to numpy array for easier computation
    bb_array = np.array(bounding_boxes_1mm)
    
    # Use 95th percentile as reference (covers most cases)
    reference_size = np.percentile(bb_array, 95, axis=0)
    
    # Try different spacings
    spacings_to_try = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    for spacing in spacings_to_try:
        # Calculate size at this spacing
        size_at_spacing = reference_size / spacing
        
        # Round up to multiples of 8
        rounded_size = np.ceil(size_at_spacing / 8) * 8
        
        # Check constraints
        if np.max(rounded_size) <= max_size_per_dim:
            # Estimate memory usage
            voxels = np.prod(rounded_size)
            memory_gb = voxels * 2 * 4 * 4 / (1024**3)  # 2 channels, float32, overhead factor
            
            if memory_gb <= max_memory_gb:
                return {
                    'spacing': spacing,
                    'size': rounded_size.astype(int).tolist(),
                    'memory_gb': memory_gb,
                    'coverage_95th_percentile': True
                }
    
    # Fallback to conservative settings
    return {
        'spacing': 2.0,
        'size': [256, 256, 128],
        'memory_gb': 8.0,
        'coverage_95th_percentile': False
    }