#!/usr/bin/env python3
"""
Hybrid Training Script for Multimodal Survival Prediction with PET/CT and clinical variables 

This script demonstrates how to train the hybrid model combining:
- Your advanced multimodal fusion strategies
- Organizer's proven deep learning + traditional ML approach

Expected Data Structure:
/Data/Yujing/HECKTOR2025/Hecktor2025/input/
├── images/
│   ├── ct/
│   │   ├── CHUM-001.mha
│   │   ├── CHUM-002.mha
│   │   └── ...
│   └── pet/
│       ├── CHUM-001.mha
│       ├── CHUM-002.mha
│       └── ...
└── HECKTOR_2025_Training_Task_2.csv  # Contains clinical data and survival outcomes

Usage:
    python hybrid_training_example.py --data_dir /path/to/hecktor/data --output_dir ./results
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
import torch
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "organizer_baselines" / "HECKTOR2025" / "Task2"))

# Import hybrid components
from hybrid_multimodal_survival import (
    HybridSurvivalModel,
    EnhancedHecktorDataset,
    enhanced_rfs_prediction_pipeline,
    process_regions_with_segmentation
)

# Import organizer's components
try:
    from task2_prognosis import (
        create_image_transforms,
        preprocess_clinical_data,
        set_random_seed,
        RANDOM_SEED,
        IMAGE_SIZE
    )
except ImportError:
    print("Warning: Organizer's components not available. Using fallbacks.")
    RANDOM_SEED = 42
    IMAGE_SIZE = (96, 96, 96)
    
    def set_random_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def create_image_transforms():
        # Fallback transform
        from torchvision.transforms import Compose, ToTensor
        return Compose([ToTensor()])
    
    def preprocess_clinical_data(df):
        # Fallback clinical processing
        return {'features': {row['PatientID']: [0] * 10 for _, row in df.iterrows()}}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Enhanced Multimodal Survival Model')
    
    parser.add_argument('--csv_path', type=str, 
                        default="/Data/Yujing/HECKTOR2025/Hecktor2025/input/HECKTOR_2025_Training_Task_2.csv",
                        help='Path to CSV file with clinical data and survival outcomes')
    parser.add_argument('--image_base_path', type=str, 
                        default="/media/yujing/800129L/Head_and_Neck/HECKTOR_Challenge/HECKTOR 2025 Task 2 Training/Task 2",
                        help='Base directory containing patient image folders')
    parser.add_argument('--output_dir', type=str, default='./hybrid_results',
                        help='Output directory for results')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--num_iterations', type=int, default=15,
                        help='Number of iterative training iterations')
    parser.add_argument('--feature_epochs_per_iteration', type=int, default=3,
                        help='Feature extractor epochs per iteration')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with reduced data')
    
    return parser.parse_args()

def load_hecktor_data(csv_path, image_base_path):
    """
    Load HECKTOR data from unified CSV and corresponding images.
    
    Args:
        csv_path: Path to HECKTOR_2025_Training_Task_2.csv
        image_base_path: Base path to image directory (contains patient folders)
    
    Returns:
        Tuple of (clinical_data, ct_images, pet_images, patient_ids, survival_data)
    """
    print(f"Loading data from {csv_path}")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} patients from CSV")
    
    # Define clinical features (8 features as per HECKTOR challenge)
    clinical_features = [
        'Age', 'Gender', 'CenterID', 'Tobacco', 'Alcohol', 
        'Performance status', 'Treatment', 'M-stage'
    ]
    
    # Initialize data containers
    clinical_data = []
    ct_images = []
    pet_images = []
    patient_ids = []
    survival_data = []
    
    # Process each patient
    for idx, row in df.iterrows():
        patient_id = row['PatientID']
        
        # Build image paths - files are in patient subfolders with specific naming
        patient_folder = os.path.join(image_base_path, patient_id)
        ct_path = os.path.join(patient_folder, f"{patient_id}__CT.nii.gz")
        pet_path = os.path.join(patient_folder, f"{patient_id}__PT.nii.gz")
        
        # Check if both image files exist
        if os.path.exists(ct_path) and os.path.exists(pet_path):
            try:
                # Load images using SimpleITK
                ct_image = sitk.ReadImage(ct_path)
                pet_image = sitk.ReadImage(pet_path)
                
                # Convert to numpy arrays
                ct_array = sitk.GetArrayFromImage(ct_image)
                pet_array = sitk.GetArrayFromImage(pet_image)
                
                # Extract clinical features
                clinical_row = []
                for feature in clinical_features:
                    value = row[feature]
                    if pd.isna(value):
                        # Handle missing values - could use median/mode imputation
                        if feature in ['Age']:
                            clinical_row.append(0.0)  # Will normalize later
                        elif feature in ['Gender', 'CenterID', 'Tobacco', 'Alcohol', 'Performance status', 'Treatment', 'M-stage']:
                            clinical_row.append(0.0)  # Use most common value or 0
                        else:
                            clinical_row.append(0.0)
                    else:
                        clinical_row.append(float(value))
                
                # Extract survival data
                relapse = row['Relapse']  # Binary outcome (0/1)
                rfs = row['RFS']  # Time to event in days
                
                # Add to containers
                clinical_data.append(clinical_row)
                ct_images.append(ct_array)
                pet_images.append(pet_array)
                patient_ids.append(patient_id)
                survival_data.append([relapse, rfs])
                
            except Exception as e:
                print(f"Error loading images for {patient_id}: {e}")
                continue
        else:
            if not os.path.exists(ct_path):
                print(f"Missing CT file for {patient_id}: {ct_path}")
            if not os.path.exists(pet_path):
                print(f"Missing PET file for {patient_id}: {pet_path}")
    
    print(f"Successfully loaded {len(clinical_data)} patients with complete data")
    
    # Convert to numpy arrays
    clinical_data = np.array(clinical_data)
    survival_data = np.array(survival_data)
    
    # Normalize clinical features
    if len(clinical_data) > 0:
        scaler = StandardScaler()
        clinical_data = scaler.fit_transform(clinical_data)
    
    return clinical_data, ct_images, pet_images, patient_ids, survival_data

def train_hybrid_model(data, args):
    """
    Train the hybrid model using cross-validation.
    
    Args:
        data: Loaded HECKTOR data
        args: Command line arguments
        
    Returns:
        Dictionary containing training results
    """
    print("Starting hybrid model training...")
    
    # Set random seed
    set_random_seed(RANDOM_SEED)
    
    # Setup device
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get clinical feature dimension
    sample_clinical = list(data['clinical_features']['features'].values())[0]
    clinical_dim = len(sample_clinical)
    
    # Cross-validation setup
    patient_ids = data['patient_ids']
    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(patient_ids)):
        print(f"\n=== Fold {fold + 1}/{args.num_folds} ===")
        
        # Split patients
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        
        print(f"Training patients: {len(train_patients)}")
        print(f"Validation patients: {len(val_patients)}")
        
        # Create datasets
        train_dataset = EnhancedHecktorDataset(data, train_patients)
        val_dataset = EnhancedHecktorDataset(data, val_patients)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Initialize hybrid model
        model = HybridSurvivalModel(
            clinical_feature_dim=clinical_dim,
            device=device,
            feature_dim=128
        )
        
        # Train model
        print("Training hybrid model...")
        try:
            model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_iterations=args.num_iterations,
                feature_epochs_per_iteration=args.feature_epochs_per_iteration
            )
            
            # Evaluate final performance
            final_c_index = model.evaluate_system(val_loader)
            
            fold_results.append({
                'fold': fold + 1,
                'final_c_index': final_c_index,
                'best_c_index': model.best_c_index,
                'train_patients': len(train_patients),
                'val_patients': len(val_patients)
            })
            
            # Save fold model
            fold_output_dir = Path(args.output_dir) / f"fold_{fold + 1}"
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'feature_extractor': model.feature_extractor.state_dict(),
                'icare_model': model.icare_model,
                'fold_results': fold_results[-1]
            }, fold_output_dir / "hybrid_model.pth")
            
            print(f"Fold {fold + 1} completed. C-index: {final_c_index:.4f}")
            
        except Exception as e:
            print(f"Error training fold {fold + 1}: {e}")
            fold_results.append({
                'fold': fold + 1,
                'final_c_index': 0.5,
                'best_c_index': 0.5,
                'error': str(e)
            })
    
    return fold_results

def evaluate_hybrid_approach(data, fold_results, args):
    """
    Evaluate the hybrid approach and generate comprehensive results.
    
    Args:
        data: Loaded HECKTOR data  
        fold_results: Results from cross-validation training
        args: Command line arguments
        
    Returns:
        Dictionary containing evaluation results
    """
    print("Evaluating hybrid approach...")
    
    # Calculate cross-validation statistics
    valid_results = [r for r in fold_results if 'error' not in r]
    
    if not valid_results:
        print("Error: No valid fold results available")
        return {'error': 'No valid folds completed'}
    
    c_indices = [r['final_c_index'] for r in valid_results]
    best_c_indices = [r['best_c_index'] for r in valid_results]
    
    evaluation_results = {
        'cross_validation_stats': {
            'mean_c_index': np.mean(c_indices),
            'std_c_index': np.std(c_indices),
            'min_c_index': np.min(c_indices),
            'max_c_index': np.max(c_indices),
            'mean_best_c_index': np.mean(best_c_indices),
            'successful_folds': len(valid_results),
            'total_folds': len(fold_results)
        },
        'fold_details': fold_results,
        'training_parameters': {
            'num_iterations': args.num_iterations,
            'feature_epochs_per_iteration': args.feature_epochs_per_iteration,
            'batch_size': args.batch_size,
            'num_patients': len(data['patient_ids']),
            'clinical_features': len(list(data['clinical_features']['features'].values())[0])
        }
    }
    
    print(f"Cross-validation results:")
    print(f"  Mean C-index: {evaluation_results['cross_validation_stats']['mean_c_index']:.4f} ± {evaluation_results['cross_validation_stats']['std_c_index']:.4f}")
    print(f"  Best C-index: {evaluation_results['cross_validation_stats']['max_c_index']:.4f}")
    print(f"  Successful folds: {evaluation_results['cross_validation_stats']['successful_folds']}/{evaluation_results['cross_validation_stats']['total_folds']}")
    
    return evaluation_results

def test_single_prediction(args):
    """
    Test the hybrid pipeline on a single patient.
    """
    print("Testing single patient prediction...")
    
    # Example clinical data
    example_clinical = {
        'Age': 65,
        'Gender': 'M',
        'Tobacco Consumption': 'Former',
        'Alcohol Consumption': 'Yes',  
        'Performance Status': 0,
        'Treatment': 'CRT',
        'M-stage': 'M0'
    }
    
    # Test paths (update these to actual test data)
    ct_path = Path(args.data_dir) / "images" / "ct" / "CHUM-001.mha"
    pet_path = Path(args.data_dir) / "images" / "pet" / "CHUM-001.mha"
    
    if not ct_path.exists() or not pet_path.exists():
        print("Warning: Test images not found. Skipping single prediction test.")
        return None
    
    try:
        results = enhanced_rfs_prediction_pipeline(
            ct_path=str(ct_path),
            pet_path=str(pet_path),
            clinical_data=example_clinical,
            model_path=None,  # No pre-trained model for this test
            save_outputs=True,
            output_dir=str(Path(args.output_dir) / "single_prediction_test")
        )
        
        print("Single prediction test completed successfully!")
        return results
        
    except Exception as e:
        print(f"Single prediction test failed: {e}")
        return None

def main():
    """Main training and evaluation pipeline."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    training_config = {
        'arguments': vars(args),
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED
    }
    
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print("=== Enhanced Multimodal Survival Prediction Training ===")
    print(f"Output directory: {output_dir}")
    print(f"Test mode: {args.test_mode}")
    
    try:
        # Load data
        print(f"Loading data from CSV: {args.csv_path}")
        print(f"Loading images from: {args.image_base_path}")
        
        clinical_data, ct_images, pet_images, patient_ids, survival_data = load_hecktor_data(
            args.csv_path, 
            args.image_base_path
        )
        
        # Package data for compatibility with existing pipeline
        data = {
            'clinical_data': clinical_data,
            'ct_images': ct_images,
            'pet_images': pet_images,
            'patient_ids': patient_ids,
            'survival_data': survival_data
        }
        
        if len(data['patient_ids']) < args.num_folds:
            print(f"Warning: Only {len(data['patient_ids'])} patients available. Reducing folds to {len(data['patient_ids'])}")
            args.num_folds = max(2, len(data['patient_ids']) // 2)
        
        # Train hybrid model
        fold_results = train_hybrid_model(data, args)
        
        # Evaluate results
        evaluation_results = evaluate_hybrid_approach(data, fold_results, args)
        
        # Save results
        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Test single prediction
        single_prediction_results = test_single_prediction(args)
        
        if single_prediction_results:
            with open(output_dir / "single_prediction_results.json", 'w') as f:
                json.dump(single_prediction_results, f, indent=2, default=str)
        
        print(f"\nTraining completed! Results saved to {output_dir}")
        print("Files created:")
        print(f"  - training_config.json: Training configuration")
        print(f"  - evaluation_results.json: Cross-validation results")
        print(f"  - fold_*/hybrid_model.pth: Trained models for each fold")
        
        if single_prediction_results:
            print(f"  - single_prediction_results.json: Single patient test")
    
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / "error_log.json", 'w') as f:
            json.dump(error_info, f, indent=2)

if __name__ == "__main__":
    main()


# RUNNING THIS 

# cd /Data/Yujing/HECKTOR2025/Hecktor2025/outcome_pred/tmp

# # Basic training with your HECKTOR data
# python hybrid_training_example.py \
#     --data_dir /Data/Yujing/HECKTOR2025/Hecktor2025/input \
#     --output_dir ./hybrid_results \
#     --num_folds 5 \
#     --use_gpu

# Quick test mode
# Test with reduced data for debugging
# python hybrid_training_example.py \
#     --data_dir /Data/Yujing/HECKTOR2025/Hecktor2025/input \
#     --output_dir ./test_results \
#     --test_mode \
#     --num_folds 2 \
#     --num_iterations 5

# Full training
# Complete training with optimized parameters
# python hybrid_training_example.py \
#     --data_dir /Data/Yujing/HECKTOR2025/Hecktor2025/input \
#     --output_dir ./full_training_results \
#     --num_folds 5 \
#     --num_iterations 20 \
#     --feature_epochs_per_iteration 5 \
#     --batch_size 4 \
#     --use_gpu