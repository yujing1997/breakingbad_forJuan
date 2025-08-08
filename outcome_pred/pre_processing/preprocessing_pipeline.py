#!/usr/bin/env python3
"""
HECKTOR 2025 Comprehensive Preprocessing Pipeline

This script implements the systematic preprocessing approach:

Step 1 (Before Pipeline - Offline):
- Resample all patients to 1mm spacing
- Crop around segmentation masks and compute bounding box shapes
- Find optimal reference bounding box size (multiple of 8)
- Determine optimal spacing based on GPU memory constraints

Step 2 (During Pipeline - Online):
- Resample to defined spacing
- Crop around segmentation mask
- Pad to defined shape for consistent CNN input
- Apply CT normalization with nnUNet parameters
- Compute weighted average of PET/CT (early fusion)

Usage:
    # Step 1: Analyze dataset and determine optimal parameters
    python preprocessing_pipeline.py --mode analyze --data_dir /path/to/data --output_config config.json
    
    # Step 2: Apply preprocessing with determined parameters
    python preprocessing_pipeline.py --mode preprocess --config config.json --input_dir /path/to/data --output_dir /path/to/output
"""

import argparse
import json
import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import logging
from dataclasses import dataclass
import sys

# Add the src directory to path for inference capabilities
src_dir = Path(__file__).parents[2] / "src"
sys.path.insert(0, str(src_dir))

from connected_components_crop import crop_around_mask, resample_volume
from normalization import CTNormalization, PETNormalization

# Import tumor segmentation inference
try:
    from inference import Segmentator
    SEGMENTATION_AVAILABLE = True
except ImportError:
    SEGMENTATION_AVAILABLE = False
    print("Warning: Tumor segmentation inference not available. Will use provided masks or create dummy masks.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    target_spacing: Tuple[float, float, float]  # Final spacing (x, y, z)
    reference_bb_size: Tuple[int, int, int]    # Reference bounding box size (x, y, z) - multiples of 8
    margin_mm: float                            # Margin around mask in mm
    gpu_memory_limit: int                       # GPU memory limit in GB
    ct_fusion_weight: float                     # Weight for CT in PET/CT fusion (0.5 or 0.75)
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            'target_spacing': self.target_spacing,
            'reference_bb_size': self.reference_bb_size,
            'margin_mm': self.margin_mm,
            'gpu_memory_limit': self.gpu_memory_limit,
            'ct_fusion_weight': self.ct_fusion_weight
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class HecktorPreprocessingPipeline:
    """Main preprocessing pipeline for HECKTOR data"""
    
    def __init__(self, normalization_constants_path: str = None, enable_tumor_segmentation: bool = False):
        """
        Initialize preprocessing pipeline
        
        Args:
            normalization_constants_path: Path to normalization constants JSON file
            enable_tumor_segmentation: Whether to enable automatic tumor segmentation inference
        """
        if normalization_constants_path is None:
            normalization_constants_path = os.path.join(os.path.dirname(__file__), 'normalization_constants.json')
        
        self.normalization_constants = self._load_normalization_constants(normalization_constants_path)
        self.ct_normalizer = CTNormalization(self.normalization_constants['0'])
        self.pet_normalizer = PETNormalization(self.normalization_constants['1'])
        
        # Initialize tumor segmentation if available and requested
        self.enable_tumor_segmentation = enable_tumor_segmentation and SEGMENTATION_AVAILABLE
        self.segmentator = None
        
        if self.enable_tumor_segmentation:
            try:
                # Set nnUNet_results environment variable
                nnunet_results_path = str(Path(__file__).parents[2] / "nnUNet_results_submission")
                if os.environ.get('nnUNet_results') is None:
                    os.environ['nnUNet_results'] = nnunet_results_path
                
                self.segmentator = Segmentator(
                    folds=0, 
                    dataset_name="Dataset002_pet_ct_noMD", 
                    tile_step_size=0.5
                )
                logger.info("Tumor segmentation inference initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize tumor segmentation: {e}")
                self.enable_tumor_segmentation = False
        
    def _load_normalization_constants(self, filepath: str) -> Dict:
        """Load normalization constants from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Handle different possible JSON structures
            if 'foreground_intensity_properties_per_channel' in data:
                return data['foreground_intensity_properties_per_channel']
            else:
                return data
        except Exception as e:
            logger.error(f"Failed to load normalization constants from {filepath}: {e}")
            raise
    
    def analyze_dataset(self, data_dir: str, mask_pattern: str = "*__GT.nii.gz") -> Dict:
        """
        Step 1: Analyze dataset to determine optimal preprocessing parameters
        
        Args:
            data_dir: Directory containing patient data
            mask_pattern: Glob pattern for segmentation masks
            
        Returns:
            Dict with analysis results and recommended parameters
        """
        logger.info("Starting dataset analysis...")
        
        # Find all mask files
        data_path = Path(data_dir)
        mask_files = list(data_path.glob(f"**/{mask_pattern}"))
        
        if not mask_files:
            raise ValueError(f"No mask files found with pattern {mask_pattern} in {data_dir}")
        
        logger.info(f"Found {len(mask_files)} mask files")
        
        # Analyze bounding boxes at 1mm spacing
        bounding_boxes_1mm = []
        original_spacings = []
        
        for mask_file in tqdm(mask_files, desc="Analyzing masks"):
            try:
                # Load mask
                mask_image = sitk.ReadImage(str(mask_file))
                original_spacings.append(mask_image.GetSpacing())
                
                # Resample to 1mm if needed
                current_spacing = mask_image.GetSpacing()
                if not np.allclose(current_spacing, [1.0, 1.0, 1.0], atol=0.1):
                    mask_image = resample_volume(
                        mask_image, 
                        new_spacing=[1.0, 1.0, 1.0],
                        interpolator=sitk.sitkNearestNeighbor
                    )
                
                # Crop around mask and get bounding box
                try:
                    _, bounding_box = crop_around_mask(mask_image, margin_mm=0.0)
                    # bounding_box format: [x, y, z, size_x, size_y, size_z]
                    bb_size = (bounding_box[3], bounding_box[4], bounding_box[5])
                    bounding_boxes_1mm.append(bb_size)
                except Exception as e:
                    logger.warning(f"Failed to process {mask_file}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed to load {mask_file}: {e}")
                continue
        
        if not bounding_boxes_1mm:
            raise ValueError("No valid bounding boxes found")
        
        # Compute statistics
        bounding_boxes_array = np.array(bounding_boxes_1mm)
        max_bb_size = np.max(bounding_boxes_array, axis=0)
        mean_bb_size = np.mean(bounding_boxes_array, axis=0)
        percentile_95_bb_size = np.percentile(bounding_boxes_array, 95, axis=0)
        
        logger.info(f"Bounding box statistics at 1mm spacing:")
        logger.info(f"  Max size: {max_bb_size}")
        logger.info(f"  Mean size: {mean_bb_size}")
        logger.info(f"  95th percentile: {percentile_95_bb_size}")
        
        # Determine optimal spacing and bounding box size
        optimal_config = self._determine_optimal_config(max_bb_size, percentile_95_bb_size)
        
        analysis_results = {
            'num_patients': len(bounding_boxes_1mm),
            'bounding_boxes_1mm': bounding_boxes_1mm,
            'max_bb_size_1mm': max_bb_size.tolist(),
            'mean_bb_size_1mm': mean_bb_size.tolist(),
            'percentile_95_bb_size_1mm': percentile_95_bb_size.tolist(),
            'original_spacings': original_spacings,
            'optimal_config': optimal_config
        }
        
        return analysis_results
    
    def _determine_optimal_config(self, max_bb_size: np.ndarray, p95_bb_size: np.ndarray) -> Dict:
        """
        Determine optimal spacing and bounding box size based on GPU constraints
        
        Args:
            max_bb_size: Maximum bounding box size at 1mm spacing
            p95_bb_size: 95th percentile bounding box size at 1mm spacing
            
        Returns:
            Dict with optimal configuration parameters
        """
        logger.info("Determining optimal preprocessing configuration...")
        
        # Use 95th percentile as reference (covers most cases without extreme outliers)
        reference_size = p95_bb_size.copy()
        
        # Try different spacings if bounding box is too large
        spacings_to_try = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        
        for spacing in spacings_to_try:
            # Calculate size at this spacing
            size_at_spacing = reference_size / spacing
            
            # Round up to multiples of 8
            rounded_size = np.ceil(size_at_spacing / 8) * 8
            
            # Check if this fits GPU memory constraints
            memory_estimate = self._estimate_gpu_memory(rounded_size)
            
            logger.info(f"Spacing {spacing}mm: size {rounded_size}, estimated GPU memory: {memory_estimate:.1f}GB")
            
            # Accept if under reasonable GPU memory limit (e.g., 10GB)
            if memory_estimate < 10.0 and np.max(rounded_size) <= 512:
                optimal_config = {
                    'target_spacing': [spacing, spacing, spacing],
                    'reference_bb_size': rounded_size.astype(int).tolist(),
                    'estimated_gpu_memory_gb': memory_estimate,
                    'margin_mm': 10.0,  # Default margin
                    'ct_fusion_weight': 0.35  # Default CT weight for fusion
                }
                
                logger.info(f"Selected optimal configuration:")
                logger.info(f"  Spacing: {spacing}mm")
                logger.info(f"  Reference BB size: {rounded_size}")
                logger.info(f"  Estimated GPU memory: {memory_estimate:.1f}GB")
                
                return optimal_config
        
        # If no spacing works, use the largest acceptable size
        logger.warning("No optimal spacing found, using conservative settings")
        return {
            'target_spacing': [2.0, 2.0, 2.0],
            'reference_bb_size': [256, 256, 128],
            'estimated_gpu_memory_gb': 8.0,
            'margin_mm': 10.0,
            'ct_fusion_weight': 0.75
        }
    
    def _estimate_gpu_memory(self, bb_size: np.ndarray) -> float:
        """
        Estimate GPU memory usage for given bounding box size
        
        Args:
            bb_size: Bounding box size (x, y, z)
            
        Returns:
            Estimated GPU memory in GB
        """
        # Rough estimation based on:
        # - 2 channels (CT + PET) 
        # - float32 (4 bytes per voxel)
        # - Batch size of 1
        # - Additional overhead for gradients, activations, etc. (factor of 4)
        
        voxels = np.prod(bb_size)
        channels = 2
        bytes_per_voxel = 4  # float32
        overhead_factor = 4  # For gradients, activations, etc.
        
        memory_bytes = voxels * channels * bytes_per_voxel * overhead_factor
        memory_gb = memory_bytes / (1024**3)
        
        return memory_gb
    
    def _perform_tumor_segmentation(self, ct_image: sitk.Image, pet_image: sitk.Image) -> sitk.Image:
        """
        Perform tumor segmentation inference using the trained nnUNet model
        
        Args:
            ct_image: CT image
            pet_image: PET image
            
        Returns:
            Segmentation mask
        """
        if not self.enable_tumor_segmentation or self.segmentator is None:
            raise ValueError("Tumor segmentation not enabled or not available")
        
        logger.info("Running tumor segmentation inference...")
        
        try:
            # The inference.py already handles the correct preprocessing order:
            # 1. Resample PET to CT coordinate system
            # 2. Crop below brain 
            # 3. Crop z-axis
            # 4. Run nnUNet inference
            # 5. Pad back to original size
            seg_mask, metadata = self.segmentator.predict(
                image_ct=ct_image,
                image_pet=pet_image,
                preprocess=True,
                return_logits=False
            )
            
            logger.info(f"Tumor segmentation completed in {metadata.get('t1_inference', 0):.2f}s")
            
            return seg_mask
            
        except Exception as e:
            logger.error(f"Tumor segmentation failed: {e}")
            raise
    
    def _ensure_coordinate_alignment(self, ct_image: sitk.Image, pet_image: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
        """
        Ensure PET and CT are in the same coordinate system by resampling PET to CT
        This should be the FIRST step before any other processing
        
        Args:
            ct_image: CT image (reference)
            pet_image: PET image (to be resampled)
            
        Returns:
            Tuple of (ct_image, resampled_pet_image)
        """
        # Import the resampling function from inference preprocessing
        sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "preprocessing"))
        from utils import resample_pet_to_ct
        
        logger.info("Resampling PET to CT coordinate system...")
        pet_resampled = resample_pet_to_ct(pet_image, ct_image)
        
        logger.info(f"CT properties: size={ct_image.GetSize()}, spacing={ct_image.GetSpacing()}")
        logger.info(f"PET resampled: size={pet_resampled.GetSize()}, spacing={pet_resampled.GetSpacing()}")
        
        return ct_image, pet_resampled
    
    def preprocess_patient(self, 
                          ct_path: str, 
                          pet_path: str, 
                          mask_path: str = None,
                          config: PreprocessingConfig = None,
                          output_dir: str = None,
                          use_tumor_segmentation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 2: Preprocess individual patient data
        
        Args:
            ct_path: Path to CT image
            pet_path: Path to PET image  
            mask_path: Path to segmentation mask (optional if using tumor segmentation)
            config: Preprocessing configuration
            output_dir: Optional output directory to save processed images
            use_tumor_segmentation: Whether to use automatic tumor segmentation inference
            
        Returns:
            Tuple of (fused_image, mask_array)
        """
        logger.info(f"Preprocessing patient: {Path(ct_path).stem}")
        
        # Load images
        ct_image = sitk.ReadImage(ct_path)
        pet_image = sitk.ReadImage(pet_path)
        
        # STEP 1: Ensure coordinate alignment (PET resampled to CT) - MUST BE FIRST
        ct_image, pet_image = self._ensure_coordinate_alignment(ct_image, pet_image)
        
        # STEP 2: Get or generate segmentation mask
        if use_tumor_segmentation and self.enable_tumor_segmentation:
            logger.info("Using automatic tumor segmentation inference...")
            mask_image = self._perform_tumor_segmentation(ct_image, pet_image)
        elif mask_path and os.path.exists(mask_path):
            logger.info(f"Loading provided segmentation mask: {mask_path}")
            mask_image = sitk.ReadImage(mask_path)
            # Ensure mask is also in the same coordinate system
            mask_image = resample_volume(
                mask_image, 
                reference_image=ct_image,
                interpolator=sitk.sitkNearestNeighbor
            )
        else:
            logger.warning("No segmentation mask provided and tumor segmentation disabled. Creating dummy mask.")
            # Create a dummy mask in the center
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
        
        # STEP 3: Resample to target spacing (now that everything is aligned)
        if config:
            ct_resampled = resample_volume(ct_image, new_spacing=config.target_spacing)
            pet_resampled = resample_volume(pet_image, new_spacing=config.target_spacing)
            mask_resampled = resample_volume(
                mask_image, 
                new_spacing=config.target_spacing,
                interpolator=sitk.sitkNearestNeighbor
            )
        else:
            # Use 1mm spacing as default
            ct_resampled = resample_volume(ct_image, new_spacing=[1.0, 1.0, 1.0])
            pet_resampled = resample_volume(pet_image, new_spacing=[1.0, 1.0, 1.0])
            mask_resampled = resample_volume(
                mask_image, 
                new_spacing=[1.0, 1.0, 1.0],
                interpolator=sitk.sitkNearestNeighbor
            )
        
        # STEP 4: Crop around mask
        margin_mm = config.margin_mm if config else 10.0
        ct_cropped, bounding_box = crop_around_mask(ct_resampled, margin_mm=margin_mm)
        pet_cropped, _ = crop_around_mask(pet_resampled, margin_mm=margin_mm)
        mask_cropped, _ = crop_around_mask(mask_resampled, margin_mm=margin_mm)
        
        # Convert to numpy arrays
        ct_array = sitk.GetArrayFromImage(ct_cropped)
        pet_array = sitk.GetArrayFromImage(pet_cropped)
        mask_array = sitk.GetArrayFromImage(mask_cropped)
        
        # STEP 5: Pad to reference size
        if config:
            ct_padded = self._pad_to_size(ct_array, config.reference_bb_size)
            pet_padded = self._pad_to_size(pet_array, config.reference_bb_size)
            mask_padded = self._pad_to_size(mask_array, config.reference_bb_size)
        else:
            # Use arrays as-is if no config provided
            ct_padded = ct_array
            pet_padded = pet_array
            mask_padded = mask_array
        
        # STEP 6: Apply normalization
        ct_normalized = self.ct_normalizer.run(ct_padded, mask_padded)
        pet_normalized = self.pet_normalizer.run(pet_padded, mask_padded)
        
        # STEP 7: Compute weighted fusion (early fusion)
        ct_weight = config.ct_fusion_weight if config else 0.35
        fused_image = self._compute_fusion(
            ct_normalized, 
            pet_normalized, 
            ct_weight
        )
        
        # Save processed images if output directory specified
        if output_dir:
            self._save_processed_data(
                fused_image, mask_padded, 
                Path(ct_path).stem, output_dir, config
            )
        
        return fused_image, mask_padded
    
    def _pad_to_size(self, array: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Pad array to target size with zeros
        
        Args:
            array: Input array (z, y, x)
            target_size: Target size (x, y, z)
            
        Returns:
            Padded array
        """
        # Convert target size from (x, y, z) to (z, y, x) for numpy
        target_zyx = (target_size[2], target_size[1], target_size[0])
        
        current_shape = array.shape
        
        # Calculate padding for each dimension
        pad_amounts = []
        for i in range(3):
            diff = target_zyx[i] - current_shape[i]
            if diff < 0:
                # If current size is larger than target, crop
                logger.warning(f"Cropping dimension {i}: {current_shape[i]} -> {target_zyx[i]}")
                array = np.take(array, range(target_zyx[i]), axis=i)
                pad_amounts.append((0, 0))
            else:
                # Pad symmetrically
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_amounts.append((pad_before, pad_after))
        
        # Apply padding
        padded_array = np.pad(array, pad_amounts, mode='constant', constant_values=0)
        
        return padded_array
    
    def _compute_fusion(self, ct_array: np.ndarray, pet_array: np.ndarray, ct_weight: float) -> np.ndarray:
        """
        Compute weighted fusion of CT and PET
        
        Args:
            ct_array: Normalized CT array
            pet_array: Normalized PET array
            ct_weight: Weight for CT (0.5 or 0.75)
            
        Returns:
            Fused array with shape (2, z, y, x)
        """
        pet_weight = 1.0 - ct_weight
        
        # Stack CT and PET as channels
        fused = np.stack([
            ct_array * ct_weight,
            pet_array * pet_weight
        ], axis=0)
        
        return fused
    
    def _save_processed_data(self, 
                           fused_image: np.ndarray, 
                           mask: np.ndarray,
                           patient_id: str, 
                           output_dir: str,
                           config: PreprocessingConfig):
        """Save processed data to output directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays
        np.save(output_path / f"{patient_id}_fused.npy", fused_image)
        np.save(output_path / f"{patient_id}_mask.npy", mask)
        
        # Save configuration
        config.save(str(output_path / f"{patient_id}_config.json"))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HECKTOR Preprocessing Pipeline')
    
    parser.add_argument('--mode', choices=['analyze', 'preprocess'], required=True,
                        help='Pipeline mode: analyze dataset or preprocess data')
    
    # Arguments for analyze mode
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing patient data (for analyze mode)')
    parser.add_argument('--mask_pattern', type=str, default="*__GT.nii.gz",
                        help='Glob pattern for segmentation masks')
    parser.add_argument('--output_config', type=str, default='preprocessing_config.json',
                        help='Output path for preprocessing configuration')
    
    # Arguments for preprocess mode
    parser.add_argument('--config', type=str,
                        help='Path to preprocessing configuration file (for preprocess mode)')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory containing patient data (for preprocess mode)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for processed data (for preprocess mode)')
    
    # General arguments
    parser.add_argument('--normalization_constants', type=str,
                        help='Path to normalization constants JSON file')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize pipeline
    pipeline = HecktorPreprocessingPipeline(args.normalization_constants)
    
    if args.mode == 'analyze':
        # Step 1: Analyze dataset
        if not args.data_dir:
            raise ValueError("--data_dir is required for analyze mode")
        
        logger.info("Starting dataset analysis...")
        analysis_results = pipeline.analyze_dataset(args.data_dir, args.mask_pattern)
        
        # Save results
        with open(args.output_config, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Create preprocessing config
        optimal_config = analysis_results['optimal_config']
        config = PreprocessingConfig(
            target_spacing=tuple(optimal_config['target_spacing']),
            reference_bb_size=tuple(optimal_config['reference_bb_size']),
            margin_mm=optimal_config['margin_mm'],
            gpu_memory_limit=int(optimal_config['estimated_gpu_memory_gb']),
            ct_fusion_weight=optimal_config['ct_fusion_weight']
        )
        
        config_path = args.output_config.replace('.json', '_config.json')
        config.save(config_path)
        
        logger.info(f"Analysis complete. Results saved to {args.output_config}")
        logger.info(f"Preprocessing config saved to {config_path}")
    
    elif args.mode == 'preprocess':
        # Step 2: Preprocess data
        if not all([args.config, args.input_dir, args.output_dir]):
            raise ValueError("--config, --input_dir, and --output_dir are required for preprocess mode")
        
        config = PreprocessingConfig.load(args.config)
        
        # Find patient data files
        input_path = Path(args.input_dir)
        ct_files = list(input_path.glob("**/*__CT.nii.gz"))
        
        logger.info(f"Found {len(ct_files)} patients to process")
        
        for ct_file in tqdm(ct_files, desc="Processing patients"):
            patient_id = ct_file.stem.replace("__CT", "")
            
            # Find corresponding files
            pet_file = ct_file.parent / f"{patient_id}__PT.nii.gz"
            mask_file = ct_file.parent / f"{patient_id}__GT.nii.gz"
            
            if not all([pet_file.exists(), mask_file.exists()]):
                logger.warning(f"Missing files for patient {patient_id}")
                continue
            
            try:
                # Preprocess patient
                fused_image, mask = pipeline.preprocess_patient(
                    str(ct_file), str(pet_file), str(mask_file),
                    config, args.output_dir
                )
                
                logger.info(f"Successfully processed {patient_id}")
                
            except Exception as e:
                logger.error(f"Failed to process {patient_id}: {e}")
                continue
        
        logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
