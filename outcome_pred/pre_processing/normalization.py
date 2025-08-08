import SimpleITK as sitk
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional


class ImageNormalization(ABC):
    """Base class for image normalization"""
    
    def __init__(self, intensityproperties: Dict, target_dtype: np.dtype = np.float32):
        """
        Initialize normalization
        
        Args:
            intensityproperties: Dictionary with intensity statistics
            target_dtype: Target data type for normalized images
        """
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype
        self.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True
    
    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Apply normalization to image
        
        Args:
            image: Input image array
            seg: Optional segmentation mask
            
        Returns:
            Normalized image array
        """
        pass


class CTNormalization(ImageNormalization):
    """CT image normalization using nnUNet-style z-score normalization with clipping"""
    
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Apply CT normalization
        
        Args:
            image: CT image array
            seg: Optional segmentation mask (not used for CT)
            
        Returns:
            Normalized CT image
        """
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        
        # Get normalization parameters
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        # Convert to target dtype
        image = image.astype(self.target_dtype, copy=False)
        
        # Clip to percentile bounds
        np.clip(image, lower_bound, upper_bound, out=image)
        
        # Z-score normalization
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        
        return image


class PETNormalization(ImageNormalization):
    """PET image normalization using nnUNet-style z-score normalization with clipping"""
    
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Apply PET normalization
        
        Args:
            image: PET image array
            seg: Optional segmentation mask (not used for PET)
            
        Returns:
            Normalized PET image
        """
        assert self.intensityproperties is not None, "PETNormalization requires intensity properties"
        
        # Get normalization parameters
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        # Convert to target dtype
        image = image.astype(self.target_dtype, copy=False)
        
        # Clip to percentile bounds
        np.clip(image, lower_bound, upper_bound, out=image)
        
        # Z-score normalization
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        
        return image


class RobustZScoreNormalization(ImageNormalization):
    """
    Alternative normalization using robust statistics (median, MAD)
    Useful when intensity properties might not be perfectly representative
    """
    
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Apply robust z-score normalization
        
        Args:
            image: Input image array
            seg: Optional segmentation mask for foreground-only normalization
            
        Returns:
            Normalized image
        """
        image = image.astype(self.target_dtype, copy=False)
        
        if seg is not None and self.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true:
            # Normalize only foreground pixels
            mask = seg > 0
            if np.any(mask):
                foreground_pixels = image[mask]
                
                # Use median and MAD for robust normalization
                median_val = np.median(foreground_pixels)
                mad_val = np.median(np.abs(foreground_pixels - median_val))
                
                # Avoid division by zero
                if mad_val > 1e-8:
                    image[mask] = (image[mask] - median_val) / (mad_val * 1.4826)  # 1.4826 is normalization factor
                else:
                    image[mask] = image[mask] - median_val
                
                # Set background to zero
                image[~mask] = 0
        else:
            # Normalize entire image
            median_val = np.median(image)
            mad_val = np.median(np.abs(image - median_val))
            
            if mad_val > 1e-8:
                image = (image - median_val) / (mad_val * 1.4826)
            else:
                image = image - median_val
        
        return image


