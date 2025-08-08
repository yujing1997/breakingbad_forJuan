# Tests for HECKTOR Preprocessing Pipeline

This directory contains test scripts for validating the preprocessing pipeline.

## Test Files

- **`test_preprocessing.py`**: Unit tests for core preprocessing functions
- **`visualize_preprocessing.py`**: Step-by-step visualization of preprocessing pipeline
- **`__init__.py`**: Test runner script

## Running Tests

### Run All Tests
```bash
cd /Data/Yujing/HECKTOR2025/Hecktor2025/outcome_pred/pre_processing/tests
python __init__.py
```

### Run Individual Tests

#### Unit Tests
```bash
cd /Data/Yujing/HECKTOR2025/Hecktor2025/outcome_pred/pre_processing/tests
python test_preprocessing.py
```

#### Visualization Test
```bash
cd /Data/Yujing/HECKTOR2025/Hecktor2025/outcome_pred/pre_processing/tests
python visualize_preprocessing.py
```

## Test Descriptions

### Unit Tests (`test_preprocessing.py`)
- Tests data loading functions
- Validates normalization constants
- Checks bounding box calculations
- Tests resampling and cropping functions
- Validates padding operations

### Visualization Test (`visualize_preprocessing.py`)
- Runs complete preprocessing pipeline on one patient
- Shows step-by-step transformations
- Generates visualization plots
- Saves intermediate results for inspection
- Helps identify issues in the pipeline

## Expected Outputs

### Unit Tests
- Console output with test results
- Pass/fail status for each component

### Visualization Test
- `{patient_id}_fused.npy`: Final preprocessed image
- `{patient_id}_mask.npy`: Corresponding mask
- `{patient_id}_summary.json`: Processing summary
- `{patient_id}_visualization.png`: Visual plots

## Troubleshooting

If tests fail:
1. Check that all dependencies are installed
2. Verify data paths are correct
3. Ensure parent directory imports work correctly
4. Check normalization_constants.json format
