# Hybrid Multimodal Survival Prediction Framework

## 🎯 Overview

This framework implements two complementary approaches for HECKTOR 2025 Task 2 survival prediction:

1. **Organizer's Direct Fusion**: Proven baseline with DenseNet121 upgrade
2. **Enhanced Multimodal**: Advanced VAE + Attention fusion strategies

Both approaches use **DenseNet121** backbone (upgraded from ResNet-18) for superior feature extraction.

## 📋 Table of Contents

- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Architecture Diagrams](#architecture-diagrams)
- [Implementation Details](#implementation-details)
- [Usage Examples](#usage-examples)
- [Performance Comparison](#performance-comparison)

---

## 🔄 Preprocessing Pipeline

### 1. Image Preprocessing Flow

```
Raw DICOM/NIfTI Images
         ↓
    Registration & Resampling
         ↓
    Intensity Normalization
         ↓
    Cropping & Resizing
         ↓
    Data Augmentation (Training)
         ↓
    Tensor Conversion
```

### 2. Detailed Preprocessing Steps

#### Step 1: Image Loading & Registration
```python
# Load CT and PET images
ct_image = sitk.ReadImage(ct_path)      # Original resolution
pet_image = sitk.ReadImage(pet_path)    # Original resolution

# Register PET to CT space (if needed)
if not same_geometry(ct_image, pet_image):
    pet_image = register_images(pet_image, ct_image)
```

#### Step 2: Intensity Normalization
```python
# CT normalization: Clip and normalize to [-1, 1]
ct_array = sitk.GetArrayFromImage(ct_image)
ct_clipped = np.clip(ct_array, -1000, 1000)  # HU clipping
ct_normalized = (ct_clipped + 1000) / 2000.0 * 2.0 - 1.0

# PET normalization: Z-score normalization
pet_array = sitk.GetArrayFromImage(pet_image)
pet_normalized = (pet_array - pet_array.mean()) / (pet_array.std() + 1e-8)
```

#### Step 3: Spatial Processing
```python
# Target size for both approaches
TARGET_SIZE = (96, 96, 96)  # [H, W, D]

# Resample to isotropic spacing (1mm³)
resampler = sitk.ResampleImageFilter()
resampler.SetSize(TARGET_SIZE)
resampler.SetOutputSpacing([1.0, 1.0, 1.0])

ct_resampled = resampler.Execute(ct_image)
pet_resampled = resampler.Execute(pet_image)
```

#### Step 4: Region-Based Cropping (Enhanced Approach Only)
```python
# For enhanced approach: segment and crop around tumors
segmentator = Segmentator()
segmentation = segmentator.predict(ct_image, pet_image)

# Find connected components
labeled_array, num_components = label(segmentation > 0)

# Crop around each component with margin
for component_id in range(1, num_components + 1):
    mask = (labeled_array == component_id)
    cropped_ct = crop_around_mask(ct_resampled, mask, margin_mm=10.0)
    cropped_pet = crop_around_mask(pet_resampled, mask, margin_mm=10.0)
```

### 3. Clinical Data Preprocessing

```python
# Clinical features preprocessing
clinical_features = [
    'Age',                    # Continuous: normalized to [0, 1]
    'Gender',                 # Binary: M=1, F=0
    'Tobacco Consumption',    # Categorical: encoded [0, 1, 2]
    'Alcohol Consumption',    # Binary: Yes=1, No=0
    'Performance Status',     # Ordinal: [0, 1, 2, 3, 4]
    'Treatment',              # Categorical: one-hot encoded
    'M-stage',               # Categorical: M0=0, M1=1
    'N-stage',               # Ordinal: [0, 1, 2, 3]
    'T-stage',               # Ordinal: [1, 2, 3, 4]
    'Overall_stage',         # Ordinal: [1, 2, 3, 4]
    'Subsite',               # Categorical: one-hot encoded
    'GTVt_volume',           # Continuous: log-normalized
    'GTVn_volume'            # Continuous: log-normalized
]

# Normalization
scaler = StandardScaler()
clinical_normalized = scaler.fit_transform(clinical_features)
```

---

## 🏗️ Architecture Diagrams

### 1. Organizer's Direct Fusion Approach

```
INPUT STAGE:
┌─────────────┐    ┌─────────────┐
│  CT Images  │    │ PET Images  │
│ [1,96,96,96]│    │ [1,96,96,96]│
└─────────────┘    └─────────────┘
       │                  │
       └────────┬─────────┘
                │
                ▼
        ┌─────────────────┐
        │   Concatenate   │
        │  [2,96,96,96]   │
        └─────────────────┘
                │
                ▼
FEATURE EXTRACTION:
        ┌─────────────────┐
        │   DenseNet121   │
        │  (2 channels)   │
        │      ↓          │
        │  [batch, 1024]  │
        └─────────────────┘
                │
                ▼
CLINICAL PROCESSING:
┌─────────────┐           ┌─────────────────┐
│Clinical Data│  ───────→ │ Neural Network  │
│ [batch, 13] │           │ 13→64→64→32     │
└─────────────┘           │  [batch, 32]    │
                         └─────────────────┘
                                │
                                ▼
FUSION STAGE:
        ┌─────────────────────────────┐
        │     Feature Fusion          │
        │ [1024 + 32] → 512 → 256     │
        │        [batch, 128]         │
        └─────────────────────────────┘
                       │
                       ▼
OUTPUT STAGE:
        ┌─────────────────────────────┐
        │      Risk Prediction        │
        │    128 → 64 → 1            │
        │     [batch, 1]             │
        └─────────────────────────────┘
```

### 2. Enhanced Multimodal Approach

```
INPUT STAGE:
┌─────────────┐              ┌─────────────┐
│  CT Images  │              │ PET Images  │
│ [1,96,96,96]│              │ [1,96,96,96]│
└─────────────┘              └─────────────┘
       │                            │
       ▼                            ▼
SEPARATE FEATURE EXTRACTION:
┌─────────────┐              ┌─────────────┐
│ DenseNet121 │              │ DenseNet121 │
│ (CT branch) │              │ (PET branch)│
│ [batch,512] │              │ [batch,512] │
└─────────────┘              └─────────────┘
       │                            │
       └────────┬───────────────────┘
                │
                ▼
ADVANCED FUSION STRATEGIES:

┌─────────────────────────────────────────────────┐
│                VAE FUSION                       │
│  ┌─────────────┐    ┌─────────────┐             │
│  │CT Encoder   │    │PET Encoder  │             │
│  │512→256→128  │    │512→256→128  │             │
│  └─────────────┘    └─────────────┘             │
│         │                  │                   │
│         └────────┬─────────┘                   │
│                  ▼                             │
│         ┌─────────────────┐                    │
│         │  μ, σ Estimation│                    │
│         │   [batch, 64]   │                    │
│         └─────────────────┘                    │
│                  │                             │
│                  ▼                             │
│         ┌─────────────────┐                    │
│         │ Reparameterize  │                    │
│         │   [batch, 64]   │                    │
│         └─────────────────┘                    │
└─────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│              ATTENTION FUSION                   │
│  ┌─────────────┐    ┌─────────────┐             │
│  │CT Projection│    │PET Projection│            │
│  │   512→64    │    │   512→64     │            │
│  └─────────────┘    └─────────────┘             │
│         │                  │                   │
│         └────────┬─────────┘                   │
│                  ▼                             │
│         ┌─────────────────┐                    │
│         │ Attention Weights│                   │
│         │   [batch, 2]    │                    │
│         └─────────────────┘                    │
│                  │                             │
│                  ▼                             │
│         ┌─────────────────┐                    │
│         │Weighted Features│                    │
│         │   [batch, 64]   │                    │
│         └─────────────────┘                    │
└─────────────────────────────────────────────────┘

CLINICAL PROCESSING:
┌─────────────┐           ┌─────────────────┐
│Clinical Data│  ───────→ │ Neural Network  │
│ [batch, 13] │           │ 13→64→64→32     │
└─────────────┘           │  [batch, 32]    │
                         └─────────────────┘

FINAL FUSION:
        ┌─────────────────────────────────────┐
        │         Combined Features           │
        │ [VAE:64 + Attention:64 + Clinical:32]│
        │        [batch, 160] → 128           │
        └─────────────────────────────────────┘
                         │
                         ▼
OUTPUT STAGE:
        ┌─────────────────────────────┐
        │      Risk Prediction        │
        │    128 → 64 → 1            │
        │     [batch, 1]             │
        └─────────────────────────────┘
```

### 3. Training Flow Comparison

```
ORGANIZER'S TRAINING FLOW:
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Load Data  │ →  │ Train DenseNet  │ →  │Extract Features │
└─────────────┘    └─────────────────┘    └─────────────────┘
                                                   │
                                                   ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Evaluate   │ ←  │Train BaggedIcare│ ←  │ Prepare Dataset │
└─────────────┘    └─────────────────┘    └─────────────────┘

ENHANCED TRAINING FLOW:
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Region Segment│ → │  Train VAE      │ →  │Train Attention  │
└─────────────┘    └─────────────────┘    └─────────────────┘
       │                    │                      │
       ▼                    ▼                      ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Crop Regions │    │VAE Loss (MSE+KL)│    │Attention Weights│
└─────────────┘    └─────────────────┘    └─────────────────┘
       │                    │                      │
       └────────────────────┼──────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Joint Feature Training │
              │  (Survival + VAE Loss)  │
              └─────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Train BaggedIcare     │
              │   (Enhanced Features)   │
              └─────────────────────────┘
```

---

## 🛠️ Implementation Details

### 1. Key Classes and Components

```python
# Core Architecture Classes
class OrganizerFeatureExtractor(nn.Module):
    """Direct CT+PET fusion with DenseNet121"""
    - imaging_backbone: DenseNet121(in_channels=2)
    - clinical_processor: 13→64→64→32
    - feature_fusion: (1024+32)→512→256→128
    - risk_head: 128→64→1

class EnhancedMultimodalFeatureExtractor(nn.Module):
    """Separate processing + advanced fusion"""
    - ct_backbone: DenseNet121(in_channels=1)
    - pet_backbone: DenseNet121(in_channels=1)
    - vae_fusion: VAEFusion(512, 512, 64)
    - attention_fusion: AttentionWeightedFusion(512, 512, 64)
    - feature_fusion: (64+64+32)→256→128

class HybridSurvivalModel(nn.Module):
    """Unified model supporting both approaches"""
    - use_organizer_approach: bool flag
    - feature_extractor: Either organizer or enhanced
    - icare_model: BaggedIcareSurvival backend
```

### 2. Loss Functions

#### Organizer's Approach
```python
# Simple survival loss
survival_loss = DeepHitLoss(ranking_weight=0.3)
total_loss = survival_loss(risk_scores, survival_times, events)
```

#### Enhanced Approach
```python
# Multi-component loss
survival_loss = DeepHitLoss(ranking_weight=0.3)
vae_loss = reconstruction_loss + kl_divergence_loss
contrastive_loss = SurvivalContrastiveLoss(margin=2.0)

total_loss = (survival_loss + 
              0.1 * vae_loss + 
              0.1 * contrastive_loss)
```

### 3. Data Flow

#### Input Formats
```python
# For Organizer's Approach
combined_images: [batch, 2, 96, 96, 96]  # CT+PET concatenated
clinical_features: [batch, 13]
survival_times: [batch]
events: [batch]

# For Enhanced Approach  
ct_images: [batch, 1, 96, 96, 96]        # Separate CT
pet_images: [batch, 1, 96, 96, 96]       # Separate PET
clinical_features: [batch, 13]
survival_times: [batch]
events: [batch]
```

#### Feature Dimensions
```python
# Organizer's Feature Flow
CT+PET [2,96,96,96] → DenseNet121 → [1024]
Clinical [13] → MLP → [32]
Combined [1024+32] → Fusion → [128]

# Enhanced Feature Flow
CT [1,96,96,96] → DenseNet121 → [512]
PET [1,96,96,96] → DenseNet121 → [512]
VAE Fusion: [512+512] → [64]
Attention Fusion: [512+512] → [64]
Clinical [13] → MLP → [32]
Combined [64+64+32] → Fusion → [128]
```

---

## 💻 Usage Examples

### 1. Quick Start - Organizer's Approach

```python
from hybrid_multimodal_survival import HybridSurvivalModel

# Initialize model with organizer's approach
model = HybridSurvivalModel(
    clinical_feature_dim=13,
    use_organizer_approach=True,  # Key flag
    device=torch.device('cuda')
)

# Training
for batch in train_loader:
    # Combine CT and PET at input level
    ct_images = batch['ct']      # [batch, 1, 96, 96, 96]
    pet_images = batch['pet']    # [batch, 1, 96, 96, 96]
    combined = torch.cat([ct_images, pet_images], dim=1)
    
    risk_scores = model(
        combined_images=combined,
        clinical_features=batch['clinical']
    )
```

### 2. Advanced Usage - Enhanced Approach

```python
# Initialize model with enhanced fusion
model = HybridSurvivalModel(
    clinical_feature_dim=13,
    use_organizer_approach=False,  # Use advanced approach
    use_vae_fusion=True,
    use_attention_fusion=True,
    device=torch.device('cuda')
)

# Training with separate modality processing
for batch in train_loader:
    risk_scores = model(
        ct_images=batch['ct'],      # Separate CT processing
        pet_images=batch['pet'],    # Separate PET processing  
        clinical_features=batch['clinical']
    )
```

### 3. Comparative Training

```python
# Compare both approaches
results = run_comparative_experiment(
    train_loader=train_loader,
    val_loader=val_loader,
    clinical_feature_dim=13,
    device=torch.device('cuda')
)

print(f"Organizer approach: {results['organizer']['val_loss']:.4f}")
print(f"Enhanced approach: {results['enhanced']['val_loss']:.4f}")
```

### 4. Hyperparameter Optimization

```python
# Using Optuna for optimization
def objective(trial):
    model = HybridSurvivalModel(
        use_organizer_approach=False  # Optimize enhanced approach
    )
    
    features, val_loss = model.train_neural_features(
        train_loader, val_loader, trial=trial
    )
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

## 📊 Performance Comparison

### 1. Expected Performance Characteristics

| Metric | Organizer's Approach | Enhanced Approach |
|--------|---------------------|-------------------|
| **Training Speed** | ⚡ 2-3x faster | 🐌 Baseline |
| **Memory Usage** | 💚 ~50% less | 📈 Baseline |
| **Parameter Count** | 💚 ~12M parameters | 📈 ~24M parameters |
| **Feature Quality** | ✅ Good baseline | 🚀 Potentially superior |
| **Overfitting Risk** | 💚 Lower | ⚠️ Moderate |
| **Interpretability** | ✅ Simpler | 🔍 More complex |

### 2. When to Use Each Approach

#### Use Organizer's Approach When:
- ✅ Limited computational resources
- ✅ Smaller datasets (< 500 patients)
- ✅ Need fast training/inference
- ✅ Want proven, stable results
- ✅ Baseline comparison needed

#### Use Enhanced Approach When:
- 🚀 Sufficient computational resources
- 🚀 Larger datasets (> 500 patients)
- 🚀 Pushing performance boundaries
- 🚀 Analyzing multimodal interactions
- 🚀 Research/publication goals

### 3. Architecture Benefits

#### Organizer's Benefits:
- **Simplicity**: Straightforward implementation
- **Stability**: Fewer failure modes
- **Efficiency**: Lower resource requirements
- **Proven**: Based on competition-winning approach

#### Enhanced Benefits:
- **Flexibility**: Separate modality processing
- **Sophistication**: Advanced fusion strategies
- **Interpretability**: Attention weights show modality importance
- **Extensibility**: Easy to add new fusion methods

---

## 🔧 Configuration Options

### 1. Model Configuration

```python
# Organizer's Configuration
organizer_config = {
    'use_organizer_approach': True,
    'clinical_feature_dim': 13,
    'feature_dim': 128,
    'imaging_backbone': 'densenet121',
    'fusion_strategy': 'direct_concatenation'
}

# Enhanced Configuration  
enhanced_config = {
    'use_organizer_approach': False,
    'clinical_feature_dim': 13,
    'feature_dim': 128,
    'vae_latent_dim': 64,
    'use_vae_fusion': True,
    'use_attention_fusion': True,
    'imaging_backbone': 'densenet121'
}
```

### 2. Training Configuration

```python
# Standard Training
training_config = {
    'epochs': 50,
    'batch_size': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'scheduler_patience': 3
}

# Hyperparameter Optimization
optuna_config = {
    'n_trials': 50,
    'cv_folds': 3,
    'pruning': True,
    'optimize_params': [
        'lr', 'weight_decay', 'ranking_weight',
        'vae_loss_weight', 'attention_dim'
    ]
}
```

### 3. Data Configuration

```python
# Preprocessing Configuration
preprocess_config = {
    'target_size': (96, 96, 96),
    'target_spacing': (1.0, 1.0, 1.0),
    'ct_clip_range': (-1000, 1000),
    'normalization': 'z_score',
    'augmentation': True,
    'region_margin_mm': 10.0
}
```

---

## 🚀 Next Steps

1. **Data Preparation**: Set up your HECKTOR dataset following the preprocessing pipeline
2. **Environment Setup**: Install dependencies (`pip install -r requirements.txt`)
3. **Model Selection**: Choose between organizer's or enhanced approach based on your resources
4. **Training**: Run comparative experiments to determine best approach for your data
5. **Optimization**: Use Optuna for hyperparameter tuning
6. **Evaluation**: Validate on test set and analyze results

---

*This framework provides a complete solution for HECKTOR 2025 Task 2, combining proven methods with advanced research techniques. Choose the approach that best fits your computational resources and performance requirements.*
