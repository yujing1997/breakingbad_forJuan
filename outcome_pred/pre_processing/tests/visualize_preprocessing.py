#!/usr/bin/env python3
"""
Visualization Script for HECKTOR Preprocessing Pipeline

This script demonstrates each step of the preprocessing pipeline component by component
using your actual data to help identify any issues.
"""

import os
import json
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from connected_components_crop import crop_around_mask, resample_volume
from normalization import CTNormalization, PETNormalization
from preprocessing_pipeline import HecktorPreprocessingPipeline

class PreprocessingVisualizer:
    """Visualize the preprocessing pipeline step by step"""
    
    def __init__(self, data_path: str, patient_id: str = "CHUM-001", 
                 enable_tumor_segmentation: bool = False,
                 use_real_tumor_segmentation: bool = False):
        """
        Initialize visualizer
        
        Args:
            data_path: Path to data directory
            patient_id: Patient ID to visualize
            enable_tumor_segmentation: Whether to enable tumor segmentation capability
            use_real_tumor_segmentation: Whether to use automatic tumor segmentation instead of provided masks
        """
        self.data_path = Path(data_path)
        self.patient_id = patient_id
        self.enable_tumor_segmentation = enable_tumor_segmentation
        self.use_real_tumor_segmentation = use_real_tumor_segmentation
        
        # Set up paths
        self.ct_path = self._find_patient_file("CT")
        self.pet_path = self._find_patient_file("PT") 
        self.mask_path = self._find_patient_file("GT")
        
        # Output directory
        self.output_dir = Path("visualization_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline with tumor segmentation capability
        self.pipeline = HecktorPreprocessingPipeline(
            enable_tumor_segmentation=self.enable_tumor_segmentation
        )
        
    def load_normalization_constants(self) -> Dict:
        """Load normalization constants"""
        norm_file = Path(__file__).parent.parent / "normalization_constants.json"
        with open(norm_file, 'r') as f:
            return json.load(f)
    
    def visualize_step_by_step(self):
        """Run complete step-by-step visualization"""
        
        print(f"\n{'='*60}")
        print(f"PREPROCESSING VISUALIZATION FOR {self.patient_id}")
        print(f"{'='*60}")
        
        # Step 0: Load raw data
        print(f"\n🔍 STEP 0: Loading Raw Data")
        raw_ct, raw_pet, raw_mask = self.step0_load_raw_data()
        
        # Step 1: Initial resampling to 1mm
        print(f"\n🔧 STEP 1: Resample to 1mm spacing")
        resampled_ct, resampled_pet, resampled_mask = self.step1_resample_to_1mm(raw_ct, raw_pet, raw_mask)
        
        # Step 2: Crop around mask
        print(f"\n✂️ STEP 2: Crop around segmentation mask")
        cropped_ct, cropped_pet, cropped_mask, bbox = self.step2_crop_around_mask(resampled_ct, resampled_pet, resampled_mask)
        
        # Step 3: Determine optimal spacing
        print(f"\n📏 STEP 3: Determine optimal spacing")
        optimal_spacing, target_size = self.step3_determine_optimal_spacing(bbox)
        
        # Step 4: Resample to optimal spacing
        print(f"\n🎯 STEP 4: Resample to optimal spacing")
        final_ct, final_pet, final_mask = self.step4_resample_to_optimal(cropped_ct, cropped_pet, cropped_mask, optimal_spacing)
        
        # Step 5: Pad to reference size
        print(f"\n📦 STEP 5: Pad to reference size") 
        padded_ct, padded_pet, padded_mask = self.step5_pad_to_reference(final_ct, final_pet, final_mask, target_size)
        
        # Step 6: Normalize images
        print(f"\n🔧 STEP 6: Apply normalization")
        normalized_ct, normalized_pet = self.step6_normalize(padded_ct, padded_pet)
        
        # Step 7: Create fusion
        print(f"\n🔀 STEP 7: Create weighted fusion")
        fused_image = self.step7_create_fusion(normalized_ct, normalized_pet)
        
        # Step 8: Save visualization
        print(f"\n💾 STEP 8: Save visualization")
        self.step8_save_visualization(fused_image, padded_mask)
        
        print(f"\n✅ Visualization complete! Check {self.output_dir}")
        
    def step0_load_raw_data(self) -> Tuple[sitk.Image, sitk.Image, Optional[sitk.Image]]:
        """Step 0: Load raw data and show properties"""
        
        # Load CT
        ct_image = sitk.ReadImage(str(self.ct_path))
        print(f"  📊 CT Properties:")
        print(f"     Size: {ct_image.GetSize()}")
        print(f"     Spacing: {ct_image.GetSpacing()}")
        print(f"     Origin: {ct_image.GetOrigin()}")
        
        # Load PET
        pet_image = sitk.ReadImage(str(self.pet_path))
        print(f"  📊 PET Properties:")
        print(f"     Size: {pet_image.GetSize()}")
        print(f"     Spacing: {pet_image.GetSpacing()}")
        print(f"     Origin: {pet_image.GetOrigin()}")
        
        # Load mask if available
        mask_image = None
        if self.mask_path and self.mask_path.exists():
            mask_image = sitk.ReadImage(str(self.mask_path))
            print(f"  📊 Mask Properties:")
            print(f"     Size: {mask_image.GetSize()}")
            print(f"     Spacing: {mask_image.GetSpacing()}")
            print(f"     Origin: {mask_image.GetOrigin()}")
        else:
            print(f"  ⚠️ No segmentation mask found - will create dummy mask")
            # Create a dummy mask in the center for demonstration
            ct_array = sitk.GetArrayFromImage(ct_image)
            dummy_mask = np.zeros_like(ct_array)
            center = [s//2 for s in ct_array.shape]
            dummy_mask[
                center[0]-10:center[0]+10,
                center[1]-20:center[1]+20, 
                center[2]-20:center[2]+20
            ] = 1
            mask_image = sitk.GetImageFromArray(dummy_mask)
            mask_image.CopyInformation(ct_image)
        
        return ct_image, pet_image, mask_image
    
    def step1_resample_to_1mm(self, ct: sitk.Image, pet: sitk.Image, mask: sitk.Image) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
        """Step 1: Resample all images to 1mm isotropic spacing"""
        
        target_spacing = [1.0, 1.0, 1.0]
        print(f"  🎯 Target spacing: {target_spacing}")
        
        # Resample CT
        ct_resampled = resample_volume(ct, new_spacing=target_spacing, interpolator=sitk.sitkBSpline)
        print(f"  📊 CT after resampling:")
        print(f"     Size: {ct_resampled.GetSize()}")
        print(f"     Spacing: {ct_resampled.GetSpacing()}")
        
        # Resample PET  
        pet_resampled = resample_volume(pet, new_spacing=target_spacing, interpolator=sitk.sitkBSpline)
        print(f"  📊 PET after resampling:")
        print(f"     Size: {pet_resampled.GetSize()}")
        print(f"     Spacing: {pet_resampled.GetSpacing()}")
        
        # Resample mask (use nearest neighbor to preserve labels)
        mask_resampled = resample_volume(mask, new_spacing=target_spacing, interpolator=sitk.sitkNearestNeighbor)
        print(f"  📊 Mask after resampling:")
        print(f"     Size: {mask_resampled.GetSize()}")
        print(f"     Spacing: {mask_resampled.GetSpacing()}")
        
        return ct_resampled, pet_resampled, mask_resampled
    
    def step2_crop_around_mask(self, ct: sitk.Image, pet: sitk.Image, mask: sitk.Image) -> Tuple[sitk.Image, sitk.Image, sitk.Image, list]:
        """Step 2: Crop around the segmentation mask"""
        
        margin_mm = 10.0
        print(f"  📏 Margin: {margin_mm}mm")
        
        # Crop mask and get bounding box
        cropped_mask, bbox = crop_around_mask(mask, margin_mm=margin_mm)
        print(f"  📦 Bounding box: {bbox}")
        print(f"     [x_min, y_min, z_min, size_x, size_y, size_z]")
        
        # Apply same cropping to CT and PET
        from connected_components_crop import sitk_crop
        cropped_ct = sitk_crop(ct, bbox)
        cropped_pet = sitk_crop(pet, bbox)
        
        print(f"  📊 After cropping:")
        print(f"     CT size: {cropped_ct.GetSize()}")
        print(f"     PET size: {cropped_pet.GetSize()}")
        print(f"     Mask size: {cropped_mask.GetSize()}")
        
        return cropped_ct, cropped_pet, cropped_mask, bbox
    
    def step3_determine_optimal_spacing(self, bbox: list) -> Tuple[list, list]:
        """Step 3: Determine optimal spacing based on bounding box size"""
        
        current_size = bbox[3:6]  # [size_x, size_y, size_z]
        print(f"  📏 Current cropped size: {current_size}")
        
        # Start with 1mm and increase if too large
        spacing_options = [1.0, 1.5, 2.0, 2.5, 3.0]
        target_max_dim = 400  # Maximum dimension for GPU memory
        
        optimal_spacing = [1.0, 1.0, 1.0]
        
        for spacing in spacing_options:
            # Calculate what the size would be at this spacing
            new_size = [int(dim / spacing) for dim in current_size]
            max_dim = max(new_size)
            
            print(f"  🧮 Spacing {spacing}mm → size {new_size} (max: {max_dim})")
            
            if max_dim <= target_max_dim:
                optimal_spacing = [spacing, spacing, spacing]
                break
        
        # Round to multiples of 8 for GPU efficiency
        final_size = [int(dim / optimal_spacing[0]) for dim in current_size]
        final_size = [((s + 7) // 8) * 8 for s in final_size]  # Round up to multiple of 8
        
        print(f"  ✅ Optimal spacing: {optimal_spacing}")
        print(f"  ✅ Target size (multiple of 8): {final_size}")
        
        return optimal_spacing, final_size
    
    def step4_resample_to_optimal(self, ct: sitk.Image, pet: sitk.Image, mask: sitk.Image, spacing: list) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
        """Step 4: Resample to optimal spacing"""
        
        print(f"  🎯 Resampling to spacing: {spacing}")
        
        ct_resampled = resample_volume(ct, new_spacing=spacing, interpolator=sitk.sitkBSpline)
        pet_resampled = resample_volume(pet, new_spacing=spacing, interpolator=sitk.sitkBSpline)  
        mask_resampled = resample_volume(mask, new_spacing=spacing, interpolator=sitk.sitkNearestNeighbor)
        
        print(f"  📊 After optimal resampling:")
        print(f"     CT size: {ct_resampled.GetSize()}")
        print(f"     PET size: {pet_resampled.GetSize()}")
        print(f"     Mask size: {mask_resampled.GetSize()}")
        
        return ct_resampled, pet_resampled, mask_resampled
    
    def step5_pad_to_reference(self, ct: sitk.Image, pet: sitk.Image, mask: sitk.Image, target_size: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step 5: Pad images to reference size"""
        
        print(f"  📦 Target size: {target_size}")
        
        # Convert to numpy arrays
        ct_array = sitk.GetArrayFromImage(ct)  # Shape: (z, y, x)
        pet_array = sitk.GetArrayFromImage(pet)
        mask_array = sitk.GetArrayFromImage(mask)
        
        current_shape = ct_array.shape
        target_shape = (target_size[2], target_size[1], target_size[0])  # Convert (x,y,z) to (z,y,x)
        
        print(f"  📊 Current shape (z,y,x): {current_shape}")
        print(f"  📊 Target shape (z,y,x): {target_shape}")
        
        # Calculate padding
        def pad_array(array, target_shape):
            pad_widths = []
            for i in range(3):
                total_pad = max(0, target_shape[i] - array.shape[i])
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_widths.append((pad_before, pad_after))
            
            return np.pad(array, pad_widths, mode='constant', constant_values=0)
        
        padded_ct = pad_array(ct_array, target_shape)
        padded_pet = pad_array(pet_array, target_shape)
        padded_mask = pad_array(mask_array, target_shape)
        
        print(f"  ✅ Padded shape: {padded_ct.shape}")
        
        return padded_ct, padded_pet, padded_mask
    
    def step6_normalize(self, ct_array: np.ndarray, pet_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Step 6: Apply nnUNet-style normalization"""
        
        # Load normalization constants
        norm_constants = self.load_normalization_constants()
        ct_props = norm_constants["foreground_intensity_properties_per_channel"]["0"]
        pet_props = norm_constants["foreground_intensity_properties_per_channel"]["1"]
        
        print(f"  🔧 CT normalization params:")
        print(f"     Mean: {ct_props['mean']:.3f}, Std: {ct_props['std']:.3f}")
        print(f"     Percentiles: {ct_props['percentile_00_5']:.1f} to {ct_props['percentile_99_5']:.1f}")
        
        print(f"  🔧 PET normalization params:")
        print(f"     Mean: {pet_props['mean']:.3f}, Std: {pet_props['std']:.3f}")
        print(f"     Percentiles: {pet_props['percentile_00_5']:.3f} to {pet_props['percentile_99_5']:.3f}")
        
        # Apply normalization
        ct_normalizer = CTNormalization(ct_props)
        pet_normalizer = PETNormalization(pet_props)
        
        # Show before stats
        print(f"  📊 Before normalization:")
        print(f"     CT range: [{ct_array.min():.1f}, {ct_array.max():.1f}]")
        print(f"     PET range: [{pet_array.min():.3f}, {pet_array.max():.3f}]")
        
        normalized_ct = ct_normalizer.run(ct_array.copy())
        normalized_pet = pet_normalizer.run(pet_array.copy())
        
        # Show after stats
        print(f"  📊 After normalization:")
        print(f"     CT range: [{normalized_ct.min():.3f}, {normalized_ct.max():.3f}]")
        print(f"     PET range: [{normalized_pet.min():.3f}, {normalized_pet.max():.3f}]")
        
        return normalized_ct, normalized_pet
    
    def step7_create_fusion(self, ct_array: np.ndarray, pet_array: np.ndarray) -> np.ndarray:
        """Step 7: Create weighted fusion"""
        
        ct_weight = 0.75
        pet_weight = 1.0 - ct_weight
        
        print(f"  ⚖️ Fusion weights: CT={ct_weight}, PET={pet_weight}")
        
        # Create fused image with 2 channels
        fused = np.stack([
            ct_array * ct_weight,    # Channel 0: Weighted CT
            pet_array * pet_weight   # Channel 1: Weighted PET
        ], axis=0)  # Shape: (2, z, y, x)
        
        print(f"  📊 Fused image shape: {fused.shape}")
        print(f"  📊 Channel 0 (CT) range: [{fused[0].min():.3f}, {fused[0].max():.3f}]")
        print(f"  📊 Channel 1 (PET) range: [{fused[1].min():.3f}, {fused[1].max():.3f}]")
        
        return fused
    
    def step8_save_visualization(self, fused_image: np.ndarray, mask: np.ndarray):
        """Step 8: Save visualization and outputs"""
        
        # Save preprocessed data
        np.save(self.output_dir / f"{self.patient_id}_fused.npy", fused_image)
        np.save(self.output_dir / f"{self.patient_id}_mask.npy", mask)
        
        # Create visualization plots
        self.create_plots(fused_image, mask)
        
        # Save summary
        summary = {
            "patient_id": self.patient_id,
            "final_shape": fused_image.shape,
            "mask_shape": mask.shape,
            "ct_range": [float(fused_image[0].min()), float(fused_image[0].max())],
            "pet_range": [float(fused_image[1].min()), float(fused_image[1].max())],
            "mask_voxels": int(np.sum(mask > 0))
        }
        
        with open(self.output_dir / f"{self.patient_id}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  💾 Saved files:")
        print(f"     {self.patient_id}_fused.npy")
        print(f"     {self.patient_id}_mask.npy") 
        print(f"     {self.patient_id}_summary.json")
        print(f"     {self.patient_id}_visualization.png")
    
    def create_plots(self, fused_image: np.ndarray, mask: np.ndarray):
        """Create visualization plots"""
        
        try:
            # Get middle slices
            z_mid = fused_image.shape[1] // 2
            y_mid = fused_image.shape[2] // 2
            x_mid = fused_image.shape[3] // 2
            
            # Set matplotlib backend for headless environments
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Preprocessing Visualization: {self.patient_id}', fontsize=16)
            
            # CT slices
            im1 = axes[0, 0].imshow(fused_image[0, z_mid, :, :], cmap='gray')
            axes[0, 0].set_title('CT - Axial')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
            
            im2 = axes[0, 1].imshow(fused_image[0, :, y_mid, :], cmap='gray')
            axes[0, 1].set_title('CT - Coronal')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            im3 = axes[0, 2].imshow(fused_image[0, :, :, x_mid], cmap='gray')
            axes[0, 2].set_title('CT - Sagittal')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # PET slices
            im4 = axes[1, 0].imshow(fused_image[1, z_mid, :, :], cmap='hot')
            axes[1, 0].set_title('PET - Axial')
            axes[1, 0].axis('off')
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            im5 = axes[1, 1].imshow(fused_image[1, :, y_mid, :], cmap='hot')
            axes[1, 1].set_title('PET - Coronal')
            axes[1, 1].axis('off')
            plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            im6 = axes[1, 2].imshow(fused_image[1, :, :, x_mid], cmap='hot')
            axes[1, 2].set_title('PET - Sagittal')
            axes[1, 2].axis('off')
            plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            # Overlay mask contours
            for i in range(2):
                for j in range(3):
                    try:
                        if j == 0:  # Axial
                            mask_slice = mask[z_mid, :, :]
                        elif j == 1:  # Coronal  
                            mask_slice = mask[:, y_mid, :]
                        else:  # Sagittal
                            mask_slice = mask[:, :, x_mid]
                        
                        # Only draw contours if mask has non-zero values
                        if np.any(mask_slice > 0):
                            axes[i, j].contour(mask_slice, levels=[0.5], colors='red', linewidths=2)
                    except Exception as e:
                        logger.warning(f"Failed to draw contour for subplot [{i}, {j}]: {e}")
                        continue
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / f"{self.patient_id}_visualization.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create visualization plots: {e}")
            logger.info("Continuing without plots - data files were still saved successfully")

def main():
    """Main function to run visualization"""
    
    # Configuration
    base_dir = "/media/yujing/800129L/Head_and_Neck/HECKTOR_Challenge/HECKTOR 2025 Task 2 Training/Task 2"
    
    # Find first available patient
    base_path = Path(base_dir)
    patient_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('CHUM')]
    
    if not patient_dirs:
        print(f"❌ No patient directories found in {base_dir}")
        return
    
    # Use first patient for visualization
    patient_dir = patient_dirs[0]
    print(f"🔍 Using patient: {patient_dir.name}")
    
    # Create visualizer
    visualizer = PreprocessingVisualizer(str(patient_dir))
    
    # Run step-by-step visualization
    visualizer.visualize_step_by_step()

if __name__ == "__main__":
    main()
