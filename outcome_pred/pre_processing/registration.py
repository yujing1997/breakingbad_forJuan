
"""
Adapted from Sebastien (Sebbers) Quetin's work on the HaN-Seg project for the Hecktor2025
project: registrations between diagnostic PT/CT images and planning CT.

"""
# -*- coding: utf-8 -*-

###################REGISTRATION FUNCTIONS###################
import os 
import inspect
from pathlib import Path
from typing import Sequence, Union
import collections

from math import ceil
import numpy as np
import SimpleITK as sitk

from hanseg.utils.utils import sitkWriteCompressedImage


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


def transform_image(ref_img, img_to_resample, transform):
    img_to_resample_arr_og = sitk.GetArrayViewFromImage(img_to_resample)
    values, counts = np.unique(img_to_resample_arr_og, return_counts=True)
    ind = np.argmax(counts)
    bg = values[ind]
    transformed_img = sitk.Resample(
        image1=img_to_resample,
        referenceImage=ref_img,
        transform=transform,
        interpolator=sitk.sitkBSpline,
        defaultPixelValue=int(bg),
        outputPixelType=img_to_resample.GetPixelID(),
    )
    return transformed_img

def transform_image_elastix(ref_img, img_to_resample, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetTransform(transform)
    resampled_image = resampler.Execute(img_to_resample)
    return resampled_image

def compute_mutual_information(img1, img2):
    # Set the fixed and moving images
    fixed_image = sitk.Cast(img1, sitk.sitkFloat32)
    moving_image = sitk.Cast(img2, sitk.sitkFloat32)

    # Set the elastix filter
    registration_method = sitk.ImageRegistrationMethod()

    # Set the metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetMetricSamplingPercentage(1)

    registration_method.SetInterpolator(sitk.sitkLinear)
    return registration_method.MetricEvaluate(fixed_image, moving_image)


def read_elastix_transform_file(file_path):
    parameters = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('(TransformParameters'):
                parameters['TransformParameters'] = list(map(float, line.strip().replace(')','').split(' ')[1:]))
            elif line.startswith('(CenterOfRotationPoint'):
                parameters['CenterOfRotationPoint'] = list(map(float, line.strip().replace(')','').split(' ')[1:]))
            elif line.startswith('(GridSpacing'):
                parameters['GridSpacing'] = list(map(float, line.strip().replace(')','').split(' ')[1:]))
            elif line.startswith('(GridOrigin'):
                parameters['GridOrigin'] = list(map(float, line.strip().replace(')','').split(' ')[1:]))
            elif line.startswith('(GridSize'):
                parameters['GridSize'] = list(map(int, map(float, line.strip().replace(')','').split(' ')[1:])))
            elif line.startswith('(Size'):
                parameters['Size'] = list(map(int, map(float, line.strip().replace(')','').split(' ')[1:])))
            elif line.startswith('(Spacing'):
                parameters['Spacing'] = list(map(float, line.strip().replace(')','').split(' ')[1:]))
            elif line.startswith('(Transform '):
                transform_type = line.strip().replace(')','').replace('"','').split(' ')[1]
                print("Transform type: ", transform_type)
                parameters['Transform'] = transform_type
    return parameters

def create_euler_transform_from_elastix(elastix_parameters):
    if elastix_parameters['Transform'] == 'EulerTransform':
        # For 3D Euler transform
        euler_transform = sitk.Euler3DTransform()
        parameters = elastix_parameters['TransformParameters']
        center = elastix_parameters['CenterOfRotationPoint']

        # Set rotation parameters (in radians)
        euler_transform.SetRotation(parameters[0], parameters[1], parameters[2])
        
        # Set translation parameters
        euler_transform.SetTranslation(parameters[3:6])
        
        # Set center of rotation
        euler_transform.SetCenter(center)
        
        return euler_transform
    else:
        raise ValueError("The provided transform is not an EulerTransform")

def create_bspline_transform_from_elastix(elastix_parameters, image, elastixImageFilter):
    if elastix_parameters['Transform'] == 'BSplineTransform':
        # Extract BSpline transform parameters
        transform_parameters = elastix_parameters['TransformParameters']

        # Create the B-spline transform
        dimension = 3 
        bspline_order = 3
        grid_physical_spacing = elastix_parameters['GridSpacing']  # A control point every 15.0mm
        image_physical_size = [
            size * spacing
            for size, spacing in zip(image.GetSize(), image.GetSpacing())
        ]
        mesh_size = [
            int(image_size / grid_spacing + 0.5)
            for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)
        ]
        # The starting mesh size will be 1/4 of the original, it will be refined by
        # the multi-resolution framework.

        direction = image.GetDirection()
        bspline_transform = sitk.BSplineTransform(dimension, bspline_order)
        bspline_transform.SetTransformDomainOrigin(list(map(float, elastixImageFilter.GetTransformParameterMap()[0]['Origin'])))
        bspline_transform.SetTransformDomainMeshSize(mesh_size)
        bspline_transform.SetTransformDomainPhysicalDimensions(image_physical_size) # [spacing*(size-1) for spacing, size in zip(grid_spacing, grid_size)])
        bspline_transform.SetTransformDomainDirection(direction)
        # The number of parameters in the transform depends on the grid size and the transform domain
        n_parameters = bspline_transform.GetNumberOfParameters()
        transform_parameters = elastixImageFilter.GetTransformParameterMap()[0]['TransformParameters']
        
        bspline_transform.SetParameters(list(map(float, transform_parameters)))

        return bspline_transform
    else:
        raise ValueError("The provided transform is not a BSplineTransform")
    
    
def get_reg_utils_dir():
    local_file_path = os.path.abspath( inspect.getfile(inspect.currentframe()))
    repo_src_path = Path(local_file_path).parents[2]
    return os.path.join(repo_src_path, "utils", "registration")
    
def get_data_dir(case_path):
    # folder before "set_1"
    data_root = Path(case_path).parents[2]
    return os.path.join(data_root, "manual_landmarks_for_evaluation")

def rigid_registration_simpleelastix(ct, mri, case_path):
        
    # We register CT to MRI to avoir sampling unnecessary points in the CT (all points from the MRI are in the CT
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(mri)
    elastixImageFilter.SetMovingImage(ct)

    reg_param_dir = get_reg_utils_dir()
    rigid_parameterMap = sitk.ReadParameterFile(os.path.join(reg_param_dir, "rigid_param_map.txt"))

    parameterMapVector = sitk.VectorOfParameterMap()
    spatial_samples = str(int(np.prod(mri.GetSize()) * 0.01))
    rigid_parameterMap["NumberOfSpatialSamples"] = [spatial_samples]
    # rigid_parameterMap["MaximumNumberOfIterations"] = ["200"]
    parameterMapVector.append(rigid_parameterMap)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    registered_ct_elastix = elastixImageFilter.GetResultImage()
    if case_path is None:
        local_file_path = os.path.abspath( inspect.getfile(inspect.currentframe()))
        dest_path = Path(local_file_path).parents[0]
    else:
        dest_path = case_path
    param_file_path = os.path.join(dest_path, "temp_elastix_euler_transform.txt")
    sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0], param_file_path)
    # Create a SimpleITK Euler transform
    elastix_parameters = read_elastix_transform_file(param_file_path)
    if case_path is None:
        #delete temp file
        os.remove(param_file_path)
    euler_transform = create_euler_transform_from_elastix(elastix_parameters)

    return euler_transform, elastixImageFilter

def non_rigid_registration_simpleelastix(ct, rigi_reg_mri, case_path, mask_mri=False):

    ### Transfer to sitk transform not working yet.
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(ct)
    elastixImageFilter.SetMovingImage(rigi_reg_mri)
    parameterMapVector = sitk.VectorOfParameterMap()
    spatial_samples = str(int(np.prod(rigi_reg_mri.GetSize()) * 0.005))
    print('If ou were to take 1% of the image, you would have: ', spatial_samples, 'samples.')
    reg_param_dir = get_reg_utils_dir()
    parameterMap = sitk.ReadParameterFile(os.path.join(reg_param_dir, "deformable_param_map.txt"))
    # parameterMap["NumberOfSpatialSamples"] = [spatial_samples]
    # parameterMap["MaximumNumberOfIterations"] = ["200"]

    if mask_mri:
        print("CREATING MASKSS")
        moving_binary_mask = sitk.BinaryThreshold(
            rigi_reg_mri,
            lowerThreshold=1,
            upperThreshold=5000,
            insideValue=1,
            outsideValue=0,
        )
        sitkWriteCompressedImage(moving_binary_mask, os.path.join(case_path, "moving_mask.seg.nrrd"))

        elastixImageFilter.SetFixedMask(moving_binary_mask)
        elastixImageFilter.SetMovingMask(moving_binary_mask)
        parameterMap["MaximumNumberOfSamplingAttempts"] = ["100"]
        parameterMap["NumberOfSpatialSamples"] = [spatial_samples]

    parameterMapVector.append(parameterMap)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastix_transform = elastixImageFilter.Execute()
    registered_rnr_mri_elastix = elastixImageFilter.GetResultImage()
    sitkWriteCompressedImage(registered_rnr_mri_elastix, os.path.join(case_path, "non_rigid_reg_mri_elastix.nii.gz"))

    output_transform_file = os.path.join(case_path, "elastix_bspline_transform.txt")
    sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0], output_transform_file)


    # Create sitk trasnform from elastix transform
    elastix_parameters = read_elastix_transform_file(output_transform_file)
    bspline_transform = create_bspline_transform_from_elastix(elastix_parameters, image=ct, elastixImageFilter=elastixImageFilter)
    sitk.WriteTransform(bspline_transform, os.path.join(case_path, "bspline_transform_from_elastix.tfm"))

    return bspline_transform, elastixImageFilter


def rigid_registration(
    fixed_image,
    moving_image,
    mask_softtissue=False,
    multi_res=True,
    transform="rigid",
    init_guess="center",
    optimizer="line_search",
    mutualinfo_bins=50,
    sampling_percent=1,
    output=None,
):

    if transform == "affine":
        tx = sitk.AffineTransform(fixed_image.GetDimension())
    else:
        assert transform == "rigid"
        tx = sitk.Euler3DTransform()
    if init_guess == "center":
        init_pos = sitk.CenteredTransformInitializerFilter.GEOMETRY
    else:
        assert init_guess == "moments"
        init_pos = sitk.CenteredTransformInitializerFilter.MOMENTS

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, tx, init_pos
    )
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    assert (
        mutualinfo_bins >= 50 and mutualinfo_bins <= 100
    ), "works best in my experience with our dataset"
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=mutualinfo_bins
    )  # 3m25 with 5000 compared to 56 s with 1000
    if mask_softtissue:
        fixed_binary_mask = sitk.BinaryThreshold(
            fixed_image,
            lowerThreshold=-900,
            upperThreshold=3000,
            insideValue=1,
            outsideValue=0,
        )
        # sitkWriteCompressedImage(fixed_binary_mask, "/scratch/student/sebastienquetin/Data/HaN-Seg/fixed_mask.seg.nrrd")
        # IF you want to print it you can use:
        # interact(display_images, fixed_image_z=(0,fixed_image.GetSize()[2]-1), moving_image_z=(0,binary_mask.GetSize()[2]-1), fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image)), moving_npa=fixed(sitk.GetArrayViewFromImage(binary_mask)));
        registration_method.SetMetricFixedMask(fixed_binary_mask)
        moving_binary_mask = sitk.BinaryThreshold(
            moving_image,
            lowerThreshold=25,
            upperThreshold=5000,
            insideValue=1,
            outsideValue=0,
        )
        # sitkWriteCompressedImage(moving_binary_mask, "/scratch/student/sebastienquetin/Data/HaN-Seg/moving_mask.seg.nrrd")
        registration_method.SetMetricMovingMask(moving_binary_mask)

    # RANDOM as opposite to a grid (REGULAR) makes the search space more flexible and can lead to better optimization
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(
        sampling_percent / 100
    )  # NO SEED IS SET since we might be running the reg multiple times

    registration_method.SetInterpolator(
        sitk.sitkLinear
    )  # sitkBSplineResamplerOrder3) sitkLinear
    # linear 0m56 seconds vs 2m22 s and same metric -0.59

    # Optimizer settings.
    assert optimizer in ["normal", "line_search", "step"]
    if optimizer == "normal":
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=5,
        )
    elif optimizer == "line_search":
        registration_method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=0.01,
            numberOfIterations=200,
            convergenceMinimumValue=1e-4,
            convergenceWindowSize=10,
            lineSearchUpperLimit=5.0,
            lineSearchMaximumIterations=200,
        )
    else:
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=100,
            relaxationFactor=0.5,
            gradientMagnitudeTolerance=1e-4,
            maximumStepSizeInPhysicalUnits=0.0,
        )

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    if multi_res:
        shrinkFactors = [4, 2, 1]
        smoothingSigmas = [2, 1, 0]
    else:
        shrinkFactors = [1]
        smoothingSigmas = [0]
    registration_method.SetShrinkFactorsPerLevel(
        shrinkFactors=shrinkFactors
    )  # [4,2,1] 3 min 4.2s -0.59 metric
    registration_method.SetSmoothingSigmasPerLevel(
        smoothingSigmas=smoothingSigmas
    )  # [2,1,0] 2.27 -0.588 with only one resoliution 2m48 -0.592 with 2 res
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    if output is not None:
        # Connect all of the observers so that we can perform plotting during registration.
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(
            sitk.sitkMultiResolutionIterationEvent, update_multires_iterations
        )
        if output == "plot":
            registration_method.AddCommand(
                sitk.sitkIterationEvent, lambda: plot_values(registration_method)
            )
        else:
            assert output == "save"
            registration_method.AddCommand(
                sitk.sitkIterationEvent,
                lambda: save_plot(
                    registration_method,
                    fixed_image,
                    moving_image,
                    initial_transform,
                    "/home/sebquet/VisionResearchLab/HN_Challenge/HaN_Challenge/Docker/src/preprocess/output/iteration_plot",
                ),
            )

    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32),
    )
    return final_transform, registration_method


def nonrigid_registration(
    fixed,
    moving,
    mesh_size=5,
    mask_softtissue=False,
    mask_mri=True,
    previous_transform=None,
    LGBS=True,
    output="plot",
    multi_res=True,
):

    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    grid_physical_spacing = [15.0, 15.0, 15.0]  # A control point every 15.0mm
    # grid_physical_spacing = [50.0, 50.0, 50.0]  # A control point every 20.0mm
    image_physical_size = [
        size * spacing
        for size, spacing in zip(fixed.GetSize(), fixed.GetSpacing())
    ]
    mesh_size = [
        int(image_size / grid_spacing + 0.5)
        for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)
    ]
    # The starting mesh size will be 1/4 of the original, it will be refined by
    # the multi-resolution framework.
    mesh_size = [int(sz / 4 + 0.5) for sz in mesh_size]
    print("mesh size: ", mesh_size)
    initial_transform = sitk.BSplineTransformInitializer(
        image1=fixed, transformDomainMeshSize=mesh_size, order=3
    )

    print(
        "Initial Number of Parameters: {0}".format(
            initial_transform.GetNumberOfParameters()
        )
    )
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    if mask_softtissue:
        binary_mask = sitk.BinaryThreshold(
            fixed,
            lowerThreshold=-900,
            upperThreshold=3000,
            insideValue=1,
            outsideValue=0,
        )
        registration_method.SetMetricFixedMask(binary_mask)
    if mask_mri:
        print("CREATING MASKSS")
        moving_binary_mask = sitk.BinaryThreshold(
            moving,
            lowerThreshold=1,
            upperThreshold=5000,
            insideValue=1,
            outsideValue=0,
        )
        # sitkWriteCompressedImage(moving_binary_mask, "/scratch/student/sebastienquetin/Data/moving_mask.seg.nrrd")
        rigid_reg_mri = transform_image_elastix(fixed, moving, previous_transform)
        reg_moving_binary_mask = sitk.BinaryThreshold(
            rigid_reg_mri,
            lowerThreshold=1,
            upperThreshold=5000,
            insideValue=1,
            outsideValue=0,
        )
        # sitkWriteCompressedImage(reg_moving_binary_mask, "/scratch/student/sebastienquetin/Data/reg_moving_mask.seg.nrrd")
        # sitkWriteCompressedImage(moving_binary_mask, "/scratch/student/sebastienquetin/Data/HaN-Seg/moving_mask.seg.nrrd")
        # It is already registered rigidly so we can use the MRI FOV for non rigid reg.

        registration_method.SetMetricMovingMask(moving_binary_mask)
        registration_method.SetMetricFixedMask(reg_moving_binary_mask)

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01, seed=42)

    # Multi-resolution framework.
    if multi_res:
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if LGBS:
        # registration_method.SetOptimizerAsLBFGSB(
        #     gradientConvergenceTolerance=1e-7,
        #     numberOfIterations=100,  # at each resolution
        #     costFunctionConvergenceFactor=1e7,  # 1e+7 for medium accuracy, 1e+12 for low accuracy, 1e+3 for high accuracy
        #     lowerBound=-10.0,  # lower bound of the parameter space (all parameters)
        #     upperBound=10.0,
        # )
        registration_method.SetOptimizerAsLBFGS2(
            solutionAccuracy=1e-2, numberOfIterations=100, deltaConvergenceTolerance=0.01
        ) 
    else:

        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=50,
            relaxationFactor=0.5,
            gradientMagnitudeTolerance=1e-6,
            maximumStepSizeInPhysicalUnits=0.0,
        )
        # registration_method.SetOptimizerAsGradientDescentLineSearch(
        #     learningRate=1.0,
        #     numberOfIterations=100,
        #     convergenceMinimumValue=1e-6,
        #     convergenceWindowSize=10)

    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetInitialTransformAsBSpline(
        initial_transform, inPlace=True, scaleFactors=[1, 2, 4] # scaleFactor is if you scale mesh size beforehand
    )

    if previous_transform is not None:
        registration_method.SetMovingInitialTransform(previous_transform)

    if output is not None:
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(
            sitk.sitkMultiResolutionIterationEvent, update_multires_iterations
        )

        if output == "plot":
            registration_method.AddCommand(
                sitk.sitkIterationEvent, lambda: plot_values(registration_method)
            )
        else:
            assert output == "save"
            registration_method.AddCommand(
                sitk.sitkIterationEvent,
                lambda: save_plot(
                    registration_method,
                    fixed,
                    moving,
                    initial_transform,
                    "/home/sebquet/VisionResearchLab/HN_Challenge/HaN_Challenge/output/iteration_plot",
                ),
            )
    print("Starting registration")
    outTx = registration_method.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
    )
    print("Finished registration")
    return outTx, registration_method


#################VISUALIZATION FUNCTIONS#################
import matplotlib.pyplot as plt


from IPython.display import clear_output


# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1, 2, figsize=(10, 8))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_npa[fixed_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title("fixed image")
    plt.axis("off")

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(moving_npa[moving_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title("moving image")
    plt.axis("off")

    plt.show()


# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis("off")
    plt.show()


# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []


# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, "r")
    plt.plot(
        multires_iterations,
        [metric_values[index] for index in multires_iterations],
        "b*",
    )
    plt.xlabel("Iteration Number", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.show()


# Callback invoked when the sitkMultiregistration_methodesolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


# Paste the two given images together. On the left will be image1 and on the right image2.
# image2 is also centered vertically in the combined image.
def write_combined_image(image1, image2, horizontal_space, file_name):
    combined_image = sitk.Image(
        (
            image1.GetWidth() + image2.GetWidth() + horizontal_space,
            max(image1.GetHeight(), image2.GetHeight()),
        ),
        image1.GetPixelID(),
        image1.GetNumberOfComponentsPerPixel(),
    )
    combined_image = sitk.Paste(
        combined_image, image1, image1.GetSize(), (0, 0), (0, 0)
    )
    combined_image = sitk.Paste(
        combined_image,
        image2,
        image2.GetSize(),
        (0, 0),
        (
            image1.GetWidth() + horizontal_space,
            round((combined_image.GetHeight() - image2.GetHeight()) / 2),
        ),
    )
    sitkWriteCompressedImage(combined_image, file_name)


# Callback invoked when the IterationEvent happens, update our data and
# save an image that includes a visualization of the registered images and
# the metric value plot.
def save_plot(registration_method, fixed, moving, transform, file_name_prefix):
    #
    # Plotting the similarity metric values, resolution changes are marked with
    # a blue star.
    #
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Plot the similarity metric values
    plt.plot(metric_values, "r")
    plt.plot(
        multires_iterations,
        [metric_values[index] for index in multires_iterations],
        "b*",
    )
    plt.xlabel("Iteration Number", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)

    # Convert the plot to a SimpleITK image (works with the agg matplotlib backend, doesn't work
    # with the default - the relevant method is canvas_tostring_rgb())
    plt.gcf().canvas.draw()
    plot_data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep="")
    plot_data = plot_data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    plot_image = sitk.GetImageFromArray(plot_data, isVector=True)

    #
    # Extract the central axial slice from the two volumes, compose it using the transformation
    # and alpha blend it.
    #
    alpha = 0.2

    central_index = round((fixed.GetSize())[2] / 2)

    moving_transformed = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelIDValue()
    )
    # Extract the central slice in xy and alpha blend them
    combined = (1.0 - alpha) * fixed[:, :, central_index] + alpha * moving_transformed[
        :, :, central_index
    ]

    # Assume the alpha blended images are isotropic and rescale intensity
    # Values so that they are in [0,255], convert the grayscale image to
    # color (r,g,b).
    combined_slices_image = sitk.Cast(sitk.RescaleIntensity(combined), sitk.sitkUInt8)
    combined_slices_image = sitk.Compose(
        combined_slices_image, combined_slices_image, combined_slices_image
    )

    write_combined_image(
        combined_slices_image,
        plot_image,
        0,
        file_name_prefix + format(len(metric_values), "03d") + ".png",
    )
