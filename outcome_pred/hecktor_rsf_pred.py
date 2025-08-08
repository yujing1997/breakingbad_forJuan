import numpy as np
import SimpleITK as sitk
import scipy
from scipy.ndimage import label
import os
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent))
from src.inference import Segmentator
from Hecktor2025.outcome_pred.pre_processing.connected_components import process_connected_components, crop_around_mask

# Define input paths: CT/PT dicom images, optionally (a subset) of planning CT, and RTDose files

def load_images(ct_path: str, pet_path: str):
    """
    Load CT and PET images from given paths.
    
    Args:
        ct_path: Path to the CT image file.
        pet_path: Path to the PET image file.
        
    Returns:
        Tuple of SimpleITK images (ct_image, pet_image).
    """
    ct_image = sitk.ReadImage(ct_path)
    pet_image = sitk.ReadImage(pet_path)
    return ct_image, pet_image

# nnU-Net autosegmentation of GTVp and GTVn from PET/CT
def segment_images(ct_image, pet_image):
    """
    Use nnU-Net to segment GTVp and GTVn from PET/CT images. (Hecktor25 Task 1)
    
    Args:
        ct_image: SimpleITK image of the CT scan
        pet_image: SimpleITK image of the PET scan
        
    Returns:
        SimpleITK image containing the segmentation mask
    """
    # Initialize the segmentator
    segmentator = Segmentator()
    
    # Perform segmentation
    segmentation = segmentator.predict(ct_image, pet_image)
    
    return segmentation 

# Post-process segmentation to separate connected components of all separate GTVp and GTVn regions, then crop to only those regions and save (temp)
def postprocess_and_crop(segmentation, ct_image, pet_image, margin_mm=10.0):
    """
    Post-process segmentation to separate connected components and crop around each region.
    
    Args:
        segmentation: SimpleITK segmentation mask
        ct_image: SimpleITK CT image
        pet_image: SimpleITK PET image
        margin_mm: Margin in millimeters to add around each cropped region
        
    Returns:
        List of dictionaries containing cropped CT/PET pairs for each connected component
    """
    # Convert segmentation to numpy array for processing
    seg_array = sitk.GetArrayFromImage(segmentation)
    
    # Find connected components
    labeled_array, num_components = label(seg_array)
    
    cropped_regions = []
    
    for cc_idx in range(1, num_components + 1):
        # Create mask for this connected component
        cc_mask_array = (labeled_array == cc_idx).astype(np.uint8)
        
        # Convert back to SimpleITK image with same properties as original
        cc_mask = sitk.GetImageFromArray(cc_mask_array)
        cc_mask.CopyInformation(segmentation)
        
        # Crop CT and PET around this connected component
        try:
            # Use the connected component mask for cropping
            cropped_ct, bbox_ct = crop_around_mask(ct_image, margin_mm=margin_mm)
            cropped_pet, bbox_pet = crop_around_mask(pet_image, margin_mm=margin_mm)
            
            # Also create a cropped version of the mask itself
            cropped_mask, _ = crop_around_mask(cc_mask, margin_mm=margin_mm)
            
            # Store the cropped regions
            cropped_regions.append({
                'component_id': cc_idx,
                'ct_image': cropped_ct,
                'pet_image': cropped_pet,
                'mask': cropped_mask,
                'original_mask': cc_mask,
                'bounding_box': bbox_ct,
                'volume_voxels': np.sum(cc_mask_array > 0)
            })
            
        except Exception as e:
            print(f"Warning: Could not crop component {cc_idx}: {e}")
            continue
    
    return cropped_regions

# Feature extraction from PT/CT and optional planning CT/RTDose files
# radiomics, or CNN or transformer-based feature extraction, end-to-end

def extract_features(ct_image, pet_image, mask=None, planning_ct=None, rt_dose=None, 
                    clinical_features=None, extract_radiomics=True):
    """
    Extract features from CT and PET images, optionally using planning CT and RTDose.
    Using the post-processing step to extract features from the segmented GTVp and GTVn regions.
    Clinical features: age, gender, smoker, alcohol, performance status, treatment, M-stage 
    
    Args:
        ct_image: SimpleITK image of the CT scan.
        pet_image: SimpleITK image of the PET scan.
        mask: Optional SimpleITK mask for feature extraction region
        planning_ct: Optional SimpleITK image of the planning CT.
        rt_dose: Optional SimpleITK image of the RTDose.
        clinical_features: Dictionary of clinical features
        extract_radiomics: Whether to extract radiomics features
        
    Returns:
        Dictionary containing extracted features
    """
    features = {}
    
    # Clinical features
    if clinical_features is not None:
        features['clinical'] = clinical_features
    
    # Basic image statistics
    if mask is not None:
        # Mask the images for feature extraction
        ct_array = sitk.GetArrayFromImage(ct_image)
        pet_array = sitk.GetArrayFromImage(pet_image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Extract basic statistics from masked regions
        ct_masked = ct_array[mask_array > 0]
        pet_masked = pet_array[mask_array > 0]
        
        features['ct_stats'] = {
            'mean': np.mean(ct_masked),
            'std': np.std(ct_masked),
            'min': np.min(ct_masked),
            'max': np.max(ct_masked),
            'median': np.median(ct_masked),
            'volume': np.sum(mask_array > 0)
        }
        
        features['pet_stats'] = {
            'mean': np.mean(pet_masked),
            'std': np.std(pet_masked),
            'min': np.min(pet_masked),
            'max': np.max(pet_masked),
            'median': np.median(pet_masked),
            'suv_max': np.max(pet_masked),
            'suv_mean': np.mean(pet_masked)
        }
    
    # Radiomics features (placeholder for PyRadiomics integration)
    if extract_radiomics and mask is not None:
        try:
            # This would require PyRadiomics installation
            # from radiomics import featureextractor
            # extractor = featureextractor.RadiomicsFeatureExtractor()
            # features['radiomics_ct'] = extractor.execute(ct_image, mask)
            # features['radiomics_pet'] = extractor.execute(pet_image, mask)
            
            # Placeholder radiomics features
            features['radiomics'] = {
                'texture_contrast': np.random.rand(),  # Replace with actual radiomics
                'texture_correlation': np.random.rand(),
                'texture_energy': np.random.rand(),
                'shape_sphericity': np.random.rand(),
                'shape_compactness': np.random.rand()
            }
        except ImportError:
            print("PyRadiomics not available. Skipping radiomics features.")
    
    # Deep learning features (placeholder for CNN/transformer features)
    features['deep_features'] = np.random.rand(512)  # Replace with actual CNN features
    
    return features

# Feature embeddings of each GTVp and GTVn region obtained to be fused, using the post-processing step to extract features from the segmented GTVp and GTVn regions.
    # attention-weighted intermedaite fusion (i.e. which modality is more important, or which lymph node is more important)
    # variational autoencoder (VAE) based intermediate fusion (learn the PT/CT merged representation)
    # late fusion (Ma June 2025 showed superiority for late fusion)

def multi_fusion(features_list):
    """
    Perform multi-modal fusion of features from different modalities.
    
    Args:
        features_list: List of feature arrays from different modalities.
        
    Returns:
        Fused feature representation.
    """
    # Placeholder for fusion logic
    fused_features = np.concatenate(features_list, axis=-1)  # Replace with actual fusion code
    return fused_features

# VAE intermediate fusion
def vae_fusion(features_ct, features_pet, latent_dim=64):
    """
    Perform VAE-based intermediate fusion of CT and PET features.
    
    Args:
        features_ct: CT feature array
        features_pet: PET feature array
        latent_dim: Dimension of the latent space
        
    Returns:
        Fused feature representation from VAE latent space
    """
    # Placeholder for VAE fusion logic
    # This would typically involve:
    # 1. Train a VAE on CT+PET feature pairs
    # 2. Encode both modalities to latent space
    # 3. Fuse in latent space (e.g., concatenation, addition, attention)
    # 4. Decode to get fused representation
    
    # Simple concatenation as placeholder
    if isinstance(features_ct, dict) and isinstance(features_pet, dict):
        # Handle dictionary features
        ct_array = np.concatenate([np.array(list(v.values())).flatten() 
                                  if isinstance(v, dict) else np.array([v]).flatten() 
                                  for v in features_ct.values()])
        pet_array = np.concatenate([np.array(list(v.values())).flatten() 
                                   if isinstance(v, dict) else np.array([v]).flatten() 
                                   for v in features_pet.values()])
        fused_features = np.concatenate([ct_array, pet_array])
    else:
        fused_features = np.concatenate([features_ct.flatten(), features_pet.flatten()])
    
    # Reduce to latent dimension (placeholder)
    if len(fused_features) > latent_dim:
        # Simple PCA-like reduction (placeholder)
        fused_features = fused_features[:latent_dim]
    
    return fused_features

# attention-weighted intermediate fusion
def attention_weighted_fusion(features_ct, features_pet, attention_dim=32):
    """
    Perform attention-weighted intermediate fusion of CT and PET features.
    Determines which modality is more important for each feature.
    
    Args:
        features_ct: CT feature array
        features_pet: PET feature array
        attention_dim: Dimension for attention mechanism
        
    Returns:
        Attention-weighted fused feature representation
    """
    # Convert features to arrays if they're dictionaries
    if isinstance(features_ct, dict):
        ct_array = np.concatenate([np.array(list(v.values())).flatten() 
                                  if isinstance(v, dict) else np.array([v]).flatten() 
                                  for v in features_ct.values()])
    else:
        ct_array = features_ct.flatten()
        
    if isinstance(features_pet, dict):
        pet_array = np.concatenate([np.array(list(v.values())).flatten() 
                                   if isinstance(v, dict) else np.array([v]).flatten() 
                                   for v in features_pet.values()])
    else:
        pet_array = features_pet.flatten()
    
    # Ensure same length
    min_len = min(len(ct_array), len(pet_array))
    ct_array = ct_array[:min_len]
    pet_array = pet_array[:min_len]
    
    # Simple attention mechanism (placeholder)
    # In practice, this would use learned attention weights
    attention_weights_ct = np.abs(ct_array) / (np.abs(ct_array) + np.abs(pet_array) + 1e-8)
    attention_weights_pet = 1 - attention_weights_ct
    
    # Apply attention weights
    fused_features = attention_weights_ct * ct_array + attention_weights_pet * pet_array
    
    return fused_features

# late fusion 
def late_fusion(predictions_list, weights=None):
    """
    Perform late fusion of predictions from different modalities.
    This happens at the prediction level rather than feature level.
    
    Args:
        predictions_list: List of prediction arrays from different models/modalities
        weights: Optional weights for each prediction
        
    Returns:
        Fused prediction
    """
    if weights is None:
        weights = np.ones(len(predictions_list)) / len(predictions_list)
    
    # Weighted average of predictions
    fused_prediction = np.average(predictions_list, axis=0, weights=weights)
    
    return fused_prediction

# survival prediciton using the fused features for recurrence-free survival (RFS) prediction using DeepSurv
def predict_survival(fused_features, clinical_data=None, model_type='deepsurv'):
    """
    Predict recurrence-free survival (RFS) using fused features.
    
    Args:
        fused_features: Fused feature array from multi-modal fusion
        clinical_data: Optional clinical features (age, gender, smoker, etc.)
        model_type: Type of survival model ('deepsurv', 'coxph', 'rsf')
        
    Returns:
        Dictionary containing survival predictions, hazard ratios, and risk scores
    """
    # Placeholder for survival prediction
    # In practice, this would use libraries like pycox, lifelines, or scikit-survival
    
    # Combine features
    if clinical_data is not None:
        # Ensure clinical_data is array-like
        if isinstance(clinical_data, dict):
            clinical_array = np.array(list(clinical_data.values()))
        else:
            clinical_array = np.array(clinical_data)
        
        # Combine imaging and clinical features
        combined_features = np.concatenate([fused_features.flatten(), clinical_array.flatten()])
    else:
        combined_features = fused_features.flatten()
    
    # Placeholder survival predictions
    survival_predictions = {
        'risk_score': np.random.rand(),  # Higher score = higher risk
        'hazard_ratio': np.random.rand() * 2 + 0.5,  # HR between 0.5 and 2.5
        'survival_probability_1year': np.random.rand() * 0.3 + 0.7,  # 70-100%
        'survival_probability_2year': np.random.rand() * 0.4 + 0.6,  # 60-100%
        'survival_probability_5year': np.random.rand() * 0.5 + 0.5,  # 50-100%
        'model_type': model_type,
        'feature_importance': np.random.rand(len(combined_features))
    }
    
    return survival_predictions

# Complete pipeline function
def predict_rfs_pipeline(ct_path, pet_path, clinical_data=None, save_outputs=False, output_dir=None):
    """
    Complete pipeline for RFS prediction from CT/PET images.
    
    Args:
        ct_path: Path to CT image
        pet_path: Path to PET image
        clinical_data: Dictionary of clinical features
        save_outputs: Whether to save intermediate outputs
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing all pipeline results
    """
    # Step 1: Load images
    print("Loading images...")
    ct_image, pet_image = load_images(ct_path, pet_path)
    
    # Step 2: Segment GTVp and GTVn
    print("Performing segmentation...")
    segmentation = segment_images(ct_image, pet_image)
    
    # Step 3: Post-process and crop
    print("Post-processing segmentation...")
    cropped_regions = postprocess_and_crop(segmentation, ct_image, pet_image)
    
    # Step 4: Extract features for each region
    print(f"Extracting features from {len(cropped_regions)} regions...")
    all_features = []
    
    for i, region in enumerate(cropped_regions):
        print(f"  Processing region {i+1}/{len(cropped_regions)}")
        
        # Extract features for this region
        ct_features = extract_features(region['ct_image'], region['pet_image'], 
                                     mask=region['mask'], clinical_features=clinical_data)
        
        # For multi-modal fusion, we need separate CT and PET features
        # This is a simplified approach - in practice you'd extract modality-specific features
        pet_features = extract_features(region['pet_image'], region['ct_image'], 
                                      mask=region['mask'])
        
        all_features.append({
            'region_id': region['component_id'],
            'ct_features': ct_features,
            'pet_features': pet_features
        })
    
    # Step 5: Multi-modal fusion
    print("Performing multi-modal fusion...")
    fused_features_list = []
    
    for features in all_features:
        # Try different fusion strategies
        vae_fused = vae_fusion(features['ct_features'], features['pet_features'])
        attention_fused = attention_weighted_fusion(features['ct_features'], features['pet_features'])
        
        fused_features_list.append({
            'vae_fusion': vae_fused,
            'attention_fusion': attention_fused
        })
    
    # Aggregate features across all regions (simple averaging)
    if fused_features_list:
        final_vae_features = np.mean([f['vae_fusion'] for f in fused_features_list], axis=0)
        final_attention_features = np.mean([f['attention_fusion'] for f in fused_features_list], axis=0)
    else:
        print("Warning: No valid regions found for feature extraction")
        final_vae_features = np.zeros(64)
        final_attention_features = np.zeros(64)
    
    # Step 6: Survival prediction
    print("Predicting survival...")
    vae_survival = predict_survival(final_vae_features, clinical_data, 'deepsurv_vae')
    attention_survival = predict_survival(final_attention_features, clinical_data, 'deepsurv_attention')
    
    # Late fusion of predictions
    late_fused_prediction = late_fusion([
        vae_survival['risk_score'],
        attention_survival['risk_score']
    ])
    
    # Compile results
    results = {
        'segmentation': segmentation,
        'cropped_regions': cropped_regions,
        'features': all_features,
        'fused_features': {
            'vae': final_vae_features,
            'attention': final_attention_features
        },
        'survival_predictions': {
            'vae_fusion': vae_survival,
            'attention_fusion': attention_survival,
            'late_fusion_risk': late_fused_prediction
        },
        'clinical_data': clinical_data
    }
    
    # Save outputs if requested
    if save_outputs and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save segmentation
        sitk.WriteImage(segmentation, os.path.join(output_dir, 'segmentation.nii.gz'))
        
        # Save results summary
        import json
        summary = {
            'num_regions': len(cropped_regions),
            'survival_predictions': results['survival_predictions'],
            'clinical_data': clinical_data
        }
        
        with open(os.path.join(output_dir, 'rfs_prediction_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    print("Pipeline completed successfully!")
    return results

# Example usage
if __name__ == "__main__":
    # Example clinical data
    example_clinical = {
        'age': 65,
        'gender': 'M',  # M/F
        'smoker': 1,    # 0/1
        'alcohol': 1,   # 0/1
        'performance_status': 0,  # ECOG 0-4
        'treatment': 'chemoradiation',
        'm_stage': 0    # 0/1
    }
    
    # Example usage
    ct_path = "/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/ct/CHUM-001.mha"
    pet_path = "/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/pet/CHUM-001.mha"
    
    # Run the complete pipeline
    results = predict_rfs_pipeline(
        ct_path=ct_path,
        pet_path=pet_path,
        clinical_data=example_clinical,
        save_outputs=True,
        output_dir="/Data/Yujing/HECKTOR2025/Hecktor2025/output/rfs_prediction"
    )




