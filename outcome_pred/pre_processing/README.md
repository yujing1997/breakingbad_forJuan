# HECKTOR 2025 Preprocessing Pipeline

This preprocessing pipeline implements the systematic approach for HECKTOR Task 2 data preparation with optimal GPU memory usage and consistent CNN input shapes.

## Overview

The pipeline consists of two main steps:

### Step 1: Dataset Analysis (Offline)
- Analyze all patients to determine optimal spacing and bounding box size
- Consider GPU memory constraints  
- Find reference bounding box size (multiples of 8)
- Generate preprocessing configuration

### Step 2: Data Preprocessing (Online)
- Resample to optimal spacing
- Crop around segmentation masks
- Pad to consistent reference size
- Apply nnUNet-style normalization
- Compute weighted PET/CT fusion

## Quick Start

### 1. Dataset Analysis

```bash
cd /Data/Yujing/HECKTOR2025/Hecktor2025/outcome_pred/pre_processing

# Analyze your dataset to find optimal parameters
python preprocessing_pipeline.py \
    --mode analyze \
    --data_dir "/media/yujing/800129L/Head_and_Neck/HECKTOR_Challenge/HECKTOR 2025 Task 2 Training/Task 2" \
    --mask_pattern "*__GT.nii.gz" \
    --output_config hecktor_analysis.json
```

This will create:
- `hecktor_analysis.json`: Full analysis results
- `hecktor_analysis_config.json`: Preprocessing configuration

### 2. Data Preprocessing  

```bash
# Preprocess all patients using determined parameters
python preprocessing_pipeline.py \
    --mode preprocess \
    --config hecktor_analysis_config.json \
    --input_dir "/media/yujing/800129L/Head_and_Neck/HECKTOR_Challenge/HECKTOR 2025 Task 2 Training/Task 2" \
    --output_dir "/Data/Yujing/HECKTOR2025/preprocessed_data"
```

### 3. Test the Pipeline

```bash
# Run all tests
cd tests && python __init__.py

# Or run individual tests
cd tests && python test_preprocessing.py
cd tests && python visualize_preprocessing.py
```

## File Structure

```
pre_processing/
├── preprocessing_pipeline.py      # Main pipeline implementation
├── connected_components_crop.py   # Cropping and resampling utilities
├── normalization.py              # nnUNet-style normalization
├── normalization_constants.json  # CT/PET normalization parameters
├── registration.py               # Image registration utilities
├── tests/                        # Test scripts
│   ├── __init__.py              # Test runner
│   ├── test_preprocessing.py    # Unit tests
│   ├── visualize_preprocessing.py # Pipeline visualization
│   └── README.md                # Test documentation
└── README.md                     # This file
```

## Configuration Parameters

The analysis step determines optimal parameters:

```json
{
    "target_spacing": [1.5, 1.5, 1.5],      // Optimal spacing in mm
    "reference_bb_size": [256, 256, 128],    // Reference bounding box (x,y,z)
    "margin_mm": 10.0,                       // Margin around mask
    "gpu_memory_limit": 8,                   // GPU memory limit in GB
    "ct_fusion_weight": 0.75                 // CT weight in fusion (0.5 or 0.75)
}
```

## Output Format

Each preprocessed patient produces:
- `{patient_id}_fused.npy`: Fused image with shape `(2, z, y, x)`
  - Channel 0: Weighted CT
  - Channel 1: Weighted PET  
- `{patient_id}_mask.npy`: Corresponding mask with shape `(z, y, x)`
- `{patient_id}_config.json`: Configuration used for this patient

## Integration with Training Pipeline

The preprocessed data can be directly used in your training pipeline:

```python
import numpy as np
from pathlib import Path

# Load preprocessed data
data_dir = Path("/Data/Yujing/HECKTOR2025/preprocessed_data")
patient_id = "CHUM-001"

fused_image = np.load(data_dir / f"{patient_id}_fused.npy")  # Shape: (2, z, y, x)
mask = np.load(data_dir / f"{patient_id}_mask.npy")          # Shape: (z, y, x)

# Ready for CNN input
print(f"Fused image shape: {fused_image.shape}")
print(f"Mask shape: {mask.shape}")
```

## Key Features

1. **Memory Optimization**: Automatically determines optimal spacing based on GPU constraints
2. **Consistent Input**: All images padded to same reference size for batch processing
3. **nnUNet Normalization**: Uses proven normalization from Task 1 segmentation
4. **Early Fusion**: Weighted combination of CT and PET modalities
5. **Robust Cropping**: Handles edge cases and different anatomical regions
6. **Multiples of 8**: Ensures efficient GPU memory alignment

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `reference_bb_size` or increase `target_spacing`
2. **Missing Masks**: Check `mask_pattern` matches your file naming
3. **Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install SimpleITK numpy scipy tqdm
   ```

### Custom Normalization

To use custom normalization constants, edit `normalization_constants.json`:

```json
{
    "foreground_intensity_properties_per_channel": {
        "0": {  // CT channel
            "mean": 37.857,
            "std": 43.913,
            "percentile_00_5": -109.0,
            "percentile_99_5": 161.0
        },
        "1": {  // PET channel  
            "mean": 6.359,
            "std": 4.196,
            "percentile_00_5": 0.814,
            "percentile_99_5": 22.657
        }
    }
}
```

## Performance Tips

1. **SSD Storage**: Use SSD for faster I/O during preprocessing
2. **Parallel Processing**: The pipeline can be easily parallelized by patient
3. **Memory Monitoring**: Monitor GPU memory usage during analysis step
4. **Batch Size**: Consider reference_bb_size when setting CNN batch size

## Next Steps

After preprocessing:
1. Update your training script to load `.npy` files instead of raw images
2. Adjust batch size based on `reference_bb_size` and available GPU memory
3. Use the fused images directly as CNN input (no additional preprocessing needed)
4. Consider the preprocessing configuration when designing your CNN architecture
