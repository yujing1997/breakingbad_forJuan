#!/usr/bin/env python3
"""
Test Enhanced Preprocessing Pipeline with Tumor Segmentation

This script tests the enhanced preprocessing pipeline that includes:
1. Proper PET-to-CT resampling order (first step)
2. Automatic tumor segmentation inference capability
3. Integration with the nnUNet segmentation model
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing_pipeline import HecktorPreprocessingPipeline, PreprocessingConfig


def test_enhanced_preprocessing():
    """Test the enhanced preprocessing pipeline"""
    
    print("🔬 Testing Enhanced Preprocessing Pipeline")
    print("="*60)
    
    # Test data paths
    data_path = "/media/yujing/800129L/Head_and_Neck/HECKTOR_Challenge/HECKTOR 2025 Task 2 Training/Task 2"
    patient_id = "CHUM-001"
    
    # File paths
    ct_path = f"{data_path}/{patient_id}/{patient_id}__CT.nii.gz"
    pet_path = f"{data_path}/{patient_id}/{patient_id}__PT.nii.gz"
    mask_path = f"{data_path}/{patient_id}/{patient_id}__GT.nii.gz"
    
    # Check if files exist
    if not all(Path(p).exists() for p in [ct_path, pet_path]):
        print("❌ Test data not found, using alternative paths...")
        ct_path = "/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/ct/CHUM-001.mha"
        pet_path = "/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/pet/CHUM-001.mha"
        mask_path = None
        
    print(f"📁 CT: {ct_path}")
    print(f"📁 PET: {pet_path}")
    print(f"📁 Mask: {mask_path}")
    print()
    
    # Test 1: Basic preprocessing without tumor segmentation
    print("🧪 TEST 1: Basic preprocessing without tumor segmentation")
    try:
        pipeline = HecktorPreprocessingPipeline(enable_tumor_segmentation=False)
        
        # Create basic config
        config = PreprocessingConfig(
            target_spacing=(1.0, 1.0, 1.0),
            reference_bb_size=(64, 64, 88),
            margin_mm=10.0,
            gpu_memory_limit=8,
            ct_fusion_weight=0.35
        )
        
        fused_image, mask = pipeline.preprocess_patient(
            ct_path=ct_path,
            pet_path=pet_path,
            mask_path=mask_path,
            config=config,
            use_tumor_segmentation=False
        )
        
        print(f"✅ Basic preprocessing successful!")
        print(f"   Fused image shape: {fused_image.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   CT channel range: [{fused_image[0].min():.3f}, {fused_image[0].max():.3f}]")
        print(f"   PET channel range: [{fused_image[1].min():.3f}, {fused_image[1].max():.3f}]")
        print()
        
    except Exception as e:
        print(f"❌ Basic preprocessing failed: {e}")
        print()
    
    # Test 2: Enhanced preprocessing with tumor segmentation
    print("🧪 TEST 2: Enhanced preprocessing with tumor segmentation")
    try:
        pipeline_enhanced = HecktorPreprocessingPipeline(enable_tumor_segmentation=True)
        
        if pipeline_enhanced.enable_tumor_segmentation:
            print("✅ Tumor segmentation capability enabled")
            
            fused_image_seg, mask_seg = pipeline_enhanced.preprocess_patient(
                ct_path=ct_path,
                pet_path=pet_path,
                mask_path=None,  # Force using tumor segmentation
                config=config,
                use_tumor_segmentation=True
            )
            
            print(f"✅ Enhanced preprocessing with tumor segmentation successful!")
            print(f"   Fused image shape: {fused_image_seg.shape}")
            print(f"   Segmented mask shape: {mask_seg.shape}")
            print(f"   Mask has {np.sum(mask_seg > 0)} non-zero voxels")
            print(f"   CT channel range: [{fused_image_seg[0].min():.3f}, {fused_image_seg[0].max():.3f}]")
            print(f"   PET channel range: [{fused_image_seg[1].min():.3f}, {fused_image_seg[1].max():.3f}]")
            
        else:
            print("⚠️ Tumor segmentation not available (missing nnUNet model or dependencies)")
            
    except Exception as e:
        print(f"❌ Enhanced preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("🎯 Test Summary:")
    print("   - Basic preprocessing: Coordinate alignment + provided/dummy masks")
    print("   - Enhanced preprocessing: Coordinate alignment + automatic tumor segmentation")
    print("   - Key improvement: PET resampled to CT BEFORE any other operations")
    print("   - Tumor segmentation: Uses trained nnUNet model for automatic GTVp/GTVn detection")


if __name__ == "__main__":
    # Ensure conda environment is activated
    print("🔧 Setting up environment...")
    
    # Activate conda environment if needed
    if "CONDA_DEFAULT_ENV" not in os.environ or os.environ["CONDA_DEFAULT_ENV"] != "hecktor25":
        print("⚠️ Please activate the hecktor25 conda environment first:")
        print("   conda activate hecktor25")
        sys.exit(1)
    
    test_enhanced_preprocessing()
