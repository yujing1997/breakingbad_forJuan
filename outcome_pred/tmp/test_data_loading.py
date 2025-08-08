#!/usr/bin/env python3
"""
Test script to verify HECKTOR data loading works correctly.
"""

import sys
import pandas as pd
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

# Import the data loading function
from hybrid_training_example import load_hecktor_data

def test_data_loading():
    """Test the HECKTOR data loading function."""
    print("=" * 60)
    print("Testing HECKTOR Data Loading")
    print("=" * 60)
    
    # Test data loading
    data_dir = "/Data/Yujing/HECKTOR2025/Hecktor2025/input"
    
    try:
        # Load data in test mode (limited patients)
        data = load_hecktor_data(data_dir, test_mode=True)
        
        print(f"\n✅ Data loading successful!")
        print(f"Number of patients loaded: {len(data['patient_ids'])}")
        print(f"Patient IDs: {data['patient_ids'][:5]}...")  # Show first 5
        
        # Check clinical features
        if data['clinical_features']['features']:
            sample_patient = list(data['clinical_features']['features'].keys())[0]
            sample_features = data['clinical_features']['features'][sample_patient]
            print(f"\nClinical features for {sample_patient}:")
            print(f"  Number of features: {len(sample_features)}")
            print(f"  Feature values: {sample_features}")
            
        # Check survival data
        if data['survival_data']:
            sample_patient = list(data['survival_data'].keys())[0]
            sample_survival = data['survival_data'][sample_patient]
            print(f"\nSurvival data for {sample_patient}:")
            print(f"  RFS time (days): {sample_survival['time']}")
            print(f"  Event (relapse): {sample_survival['event']}")
            
        # Check images
        if data['images']:
            sample_patient = list(data['images'].keys())[0]
            sample_image = data['images'][sample_patient]
            print(f"\nImage data for {sample_patient}:")
            print(f"  Image shape: {sample_image.shape}")
            print(f"  Image type: {type(sample_image)}")
            
        print(f"\n🎯 All data components loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_direct():
    """Test direct CSV loading to understand the format."""
    print("\n" + "=" * 60)
    print("Testing Direct CSV Loading")
    print("=" * 60)
    
    csv_path = "/Data/Yujing/HECKTOR2025/Hecktor2025/input/HECKTOR_2025_Training_Task_2.csv"
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nMissing values per column:")
        print(df.isnull().sum())
        
        print(f"\nUnique values in categorical columns:")
        categorical_cols = ['Gender', 'Tobacco Consumption', 'Alcohol Consumption', 
                          'Performance Status', 'Treatment', 'M-stage']
        for col in categorical_cols:
            if col in df.columns:
                unique_vals = df[col].unique()
                print(f"  {col}: {unique_vals}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV loading failed: {e}")
        return False

if __name__ == "__main__":
    print("HECKTOR Data Loading Test")
    
    # Test direct CSV loading first
    csv_success = test_csv_direct()
    
    if csv_success:
        # Test integrated data loading
        data_success = test_data_loading()
        
        if data_success:
            print(f"\n🎉 All tests passed! Ready to run hybrid training.")
        else:
            print(f"\n⚠️ Data loading test failed. Check the implementation.")
    else:
        print(f"\n⚠️ CSV loading test failed. Check the file path.")
