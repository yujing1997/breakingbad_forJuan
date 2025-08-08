#!/usr/bin/env python3
"""
Test script for HECKTOR preprocessing pipeline

This script tests the preprocessing pipeline components:
1. Data loading and normalization
2. Bounding box analysis
3. Resampling and cropping
4. Padding and fusion
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Add preprocessing path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing_pipeline import HecktorPreprocessingPipeline, PreprocessingConfig
from connected_components_crop import crop_around_mask, resample_volume
from normalization import CTNormalization, PETNormalization

def create_test_data(output_dir: str):
    """Create synthetic test data for pipeline testing"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating synthetic test data...")
    
    # Create synthetic CT image (256x256x128)
    ct_array = np.random.randint(-500, 1000, (128, 256, 256), dtype=np.int16)
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetSpacing([1.5, 1.5, 2.0])  # Non-isotropic spacing
    ct_image.SetOrigin([0, 0, 0])
    
    # Create synthetic PET image
    pet_array = np.random.exponential(2.0, (128, 256, 256)).astype(np.float32)
    pet_image = sitk.GetImageFromArray(pet_array)
    pet_image.SetSpacing([1.5, 1.5, 2.0])
    pet_image.SetOrigin([0, 0, 0])
    
    # Create synthetic mask (tumor in center)
    mask_array = np.zeros((128, 256, 256), dtype=np.uint8)
    # Create a spherical tumor
    center = (64, 128, 128)
    radius = 30
    for z in range(128):
        for y in range(256):
            for x in range(256):
                dist = np.sqrt((z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2)
                if dist <= radius:
                    mask_array[z, y, x] = 1
    
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.SetSpacing([1.5, 1.5, 2.0])
    mask_image.SetOrigin([0, 0, 0])
    
    # Save test data
    sitk.WriteImage(ct_image, str(output_path / "test_patient__CT.nii.gz"))
    sitk.WriteImage(pet_image, str(output_path / "test_patient__PT.nii.gz"))
    sitk.WriteImage(mask_image, str(output_path / "test_patient__GT.nii.gz"))
    
    print(f"Test data created in {output_dir}")
    return str(output_path)

def test_normalization():
    """Test normalization classes"""
    print("\n=== Testing Normalization ===")
    
    # Load normalization constants
    norm_path = Path(__file__).parent.parent / "normalization_constants.json"
    pipeline = HecktorPreprocessingPipeline(str(norm_path))
    
    # Create test arrays
    ct_array = np.random.randint(-500, 1000, (64, 64, 64), dtype=np.int16).astype(np.float32)
    pet_array = np.random.exponential(2.0, (64, 64, 64)).astype(np.float32)
    mask_array = np.ones((64, 64, 64), dtype=np.uint8)
    
    # Test CT normalization
    ct_normalized = pipeline.ct_normalizer.run(ct_array, mask_array)
    print(f"CT normalized: shape={ct_normalized.shape}, mean={ct_normalized.mean():.3f}, std={ct_normalized.std():.3f}")
    
    # Test PET normalization  
    pet_normalized = pipeline.pet_normalizer.run(pet_array, mask_array)
    print(f"PET normalized: shape={pet_normalized.shape}, mean={pet_normalized.mean():.3f}, std={pet_normalized.std():.3f}")
    
    print("✓ Normalization test passed")

def test_resampling_and_cropping():
    """Test resampling and cropping functions"""
    print("\n=== Testing Resampling and Cropping ===")
    
    # Create test image
    array = np.random.rand(50, 100, 100)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing([2.0, 1.5, 1.5])
    
    print(f"Original: shape={image.GetSize()}, spacing={image.GetSpacing()}")
    
    # Test resampling
    resampled = resample_volume(image, new_spacing=[1.0, 1.0, 1.0])
    print(f"Resampled: shape={resampled.GetSize()}, spacing={resampled.GetSpacing()}")
    
    # Create mask for cropping
    mask_array = np.zeros_like(array)
    mask_array[20:40, 30:70, 30:70] = 1
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(image)
    
    # Test cropping
    cropped, bbox = crop_around_mask(mask_image, margin_mm=5.0)
    print(f"Cropped: shape={cropped.GetSize()}, bbox={bbox}")
    
    print("✓ Resampling and cropping test passed")

def test_full_pipeline():
    """Test the full preprocessing pipeline"""
    print("\n=== Testing Full Pipeline ===")
    
    # Create test data
    test_data_dir = create_test_data("/tmp/hecktor_test_data")
    
    try:
        # Initialize pipeline
        norm_path = Path(__file__).parent.parent / "normalization_constants.json"
        pipeline = HecktorPreprocessingPipeline(str(norm_path))
        
        # Test dataset analysis
        print("Testing dataset analysis...")
        analysis_results = pipeline.analyze_dataset(test_data_dir, "*__GT.nii.gz")
        
        print(f"Analysis results:")
        print(f"  Number of patients: {analysis_results['num_patients']}")
        print(f"  Max BB size (1mm): {analysis_results['max_bb_size_1mm']}")
        print(f"  Optimal config: {analysis_results['optimal_config']}")
        
        # Create config from analysis
        optimal_config = analysis_results['optimal_config']
        config = PreprocessingConfig(
            target_spacing=tuple(optimal_config['target_spacing']),
            reference_bb_size=tuple(optimal_config['reference_bb_size']),
            margin_mm=optimal_config['margin_mm'],
            gpu_memory_limit=int(optimal_config['estimated_gpu_memory_gb']),
            ct_fusion_weight=optimal_config['ct_fusion_weight']
        )
        
        # Test individual patient preprocessing
        print("Testing patient preprocessing...")
        ct_path = f"{test_data_dir}/test_patient__CT.nii.gz"
        pet_path = f"{test_data_dir}/test_patient__PT.nii.gz"
        mask_path = f"{test_data_dir}/test_patient__GT.nii.gz"
        
        fused_image, mask = pipeline.preprocess_patient(
            ct_path, pet_path, mask_path, config
        )
        
        print(f"Preprocessing results:")
        print(f"  Fused image shape: {fused_image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Expected shape: (2, {config.reference_bb_size[2]}, {config.reference_bb_size[1]}, {config.reference_bb_size[0]})")
        
        # Verify output shape matches config
        expected_shape = (2, config.reference_bb_size[2], config.reference_bb_size[1], config.reference_bb_size[0])
        if fused_image.shape == expected_shape:
            print("✓ Output shape matches configuration")
        else:
            print(f"✗ Shape mismatch: got {fused_image.shape}, expected {expected_shape}")
        
        print("✓ Full pipeline test passed")
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
            print(f"Cleaned up test data: {test_data_dir}")

def main():
    """Run all tests"""
    print("HECKTOR Preprocessing Pipeline Tests")
    print("=" * 50)
    
    try:
        test_normalization()
        test_resampling_and_cropping() 
        test_full_pipeline()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
