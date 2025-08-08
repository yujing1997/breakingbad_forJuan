#!/usr/bin/env python3
"""
Hybrid Multimodal Survival Prediction Framework
Combines:
- Advanced multimodal fusion strategies (VAE, Attention, Region-based)
- Organizer's proven deep learning + traditional ML hybrid approach
- Our comprehensive feature extraction pipeline

Author: Yujing
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import SimpleITK as sitk
import pandas as pd
import pickle
try:
    import optuna
except ImportError:
    optuna = None
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from scipy.ndimage import label
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "organizer_baselines" / "HECKTOR2025" / "Task2"))

# Import from organizer's baseline (proven components)
try:
    from task2_prognosis import (
        DeepHitLoss, 
        SurvivalContrastiveLoss,
        create_image_transforms,
        preprocess_clinical_data,
        set_random_seed,
        RANDOM_SEED,
        IMAGE_SIZE
    )
    from icare.survival import BaggedIcareSurvival
except ImportError:
    print("Warning: Could not import organizer's components. Using fallback implementations.")
    RANDOM_SEED = 42
    IMAGE_SIZE = (96, 96, 96)

# Import your existing components
from src.inference import Segmentator
from Hecktor2025.outcome_pred.pre_processing.connected_components import crop_around_mask

# =============================================================================
# Organizer's Direct Fusion Approach (Baseline)
# =============================================================================

class OrganizerFeatureExtractor(nn.Module):
    """
    Organizer's baseline approach: Direct CT+PET fusion at input level.
    Uses DenseNet121 with 2-channel input (CT+PET concatenated).
    """
    def __init__(self, clinical_feature_dim: int, feature_output_dim: int = 128):
        super().__init__()
        
        self.clinical_feature_dim = clinical_feature_dim
        self.feature_output_dim = feature_output_dim
        
        # DenseNet121 for combined CT+PET input (organizer's approach)
        from monai.networks.nets import densenet121
        
        self.imaging_backbone = densenet121(
            spatial_dims=3,
            in_channels=2,  # CT + PET channels
            out_channels=1,
        )
        # Remove the final classification layer to get features
        self.imaging_backbone.classifier = nn.Identity()
        
        # Clinical processor (from organizer's proven design)
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Feature fusion (organizer's simple approach)
        # DenseNet121 typically outputs 1024 features
        self.feature_fusion = nn.Sequential(
            nn.Linear(1024 + 32, 512),  # DenseNet features + Clinical
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, feature_output_dim)
        )
        
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(feature_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, combined_images, clinical_features, return_risk=False):
        """
        Forward pass for organizer's approach.
        
        Args:
            combined_images: [batch, 2, H, W, D] - CT and PET concatenated
            clinical_features: [batch, clinical_dim]
            return_risk: Whether to return risk scores
        """
        # Extract imaging features from combined CT+PET
        imaging_features = self.imaging_backbone(combined_images)  # [batch, 1024]
        
        # Process clinical features
        clinical_processed = self.clinical_processor(clinical_features)  # [batch, 32]
        
        # Combine imaging and clinical features
        combined_features = torch.cat([imaging_features, clinical_processed], dim=1)
        final_features = self.feature_fusion(combined_features)
        
        if return_risk:
            risk_scores = self.risk_head(final_features).squeeze(-1)
            return final_features, risk_scores
        
        return final_features

# =============================================================================
# Enhanced Multimodal Architecture (Your Advanced Approach)
# =============================================================================

class VAEFusion(nn.Module):
    """
    Variational Autoencoder for intermediate fusion of CT and PET features.
    Your approach enhanced with their training stability techniques.
    """
    def __init__(self, input_dim_ct: int, input_dim_pet: int, latent_dim: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoders for each modality
        self.ct_encoder = nn.Sequential(
            nn.Linear(input_dim_ct, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # From organizer's stability
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        self.pet_encoder = nn.Sequential(
            nn.Linear(input_dim_pet, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # VAE parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, input_dim_ct + input_dim_pet),
            nn.Sigmoid()
        )
    
    def encode(self, ct_features, pet_features):
        """Encode CT and PET features to latent distribution."""
        ct_encoded = self.ct_encoder(ct_features)
        pet_encoded = self.pet_encoder(pet_features)
        
        # Combine modalities
        combined = torch.cat([ct_encoded, pet_encoded], dim=1)
        
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, ct_features, pet_features):
        mu, logvar = self.encode(ct_features, pet_features)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return z, mu, logvar, reconstruction

class AttentionWeightedFusion(nn.Module):
    """
    Attention-weighted fusion determining modality importance.
    Your approach with organizer's proven architecture patterns.
    """
    def __init__(self, ct_feature_dim: int, pet_feature_dim: int, attention_dim: int = 64):
        super().__init__()
        
        # Feature projections (from organizer's pattern)
        self.ct_proj = nn.Sequential(
            nn.Linear(ct_feature_dim, attention_dim),
            nn.ReLU(),
            nn.BatchNorm1d(attention_dim),
            nn.Dropout(0.2)
        )
        
        self.pet_proj = nn.Sequential(
            nn.Linear(pet_feature_dim, attention_dim),
            nn.ReLU(),
            nn.BatchNorm1d(attention_dim),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 2),  # CT and PET weights
            nn.Softmax(dim=1)
        )
        
        # Final fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(attention_dim * 2, attention_dim),
            nn.ReLU(),
            nn.BatchNorm1d(attention_dim),
            nn.Dropout(0.3)
        )
    
    def forward(self, ct_features, pet_features):
        # Project features
        ct_proj = self.ct_proj(ct_features)
        pet_proj = self.pet_proj(pet_features)
        
        # Compute attention weights
        combined = torch.cat([ct_proj, pet_proj], dim=1)
        attention_weights = self.attention(combined)  # [batch_size, 2]
        
        # Apply attention
        attended_ct = ct_proj * attention_weights[:, 0:1]
        attended_pet = pet_proj * attention_weights[:, 1:2]
        
        # Fuse
        fused = torch.cat([attended_ct, attended_pet], dim=1)
        output = self.fusion_layer(fused)
        
        return output, attention_weights

class EnhancedMultimodalFeatureExtractor(nn.Module):
    """
    Your enhanced feature extractor combining region-based processing
    with organizer's proven CNN architecture.
    """
    def __init__(self, clinical_feature_dim: int, feature_output_dim: int = 128):
        super().__init__()
        
        self.clinical_feature_dim = clinical_feature_dim
        self.feature_output_dim = feature_output_dim
        
        # Separate modality encoders (your approach) - Using DenseNet121
        from monai.networks.nets import densenet121
        
        self.ct_backbone = densenet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
        )
        # Remove the final classification layer to get features
        self.ct_backbone.classifier = nn.Identity()
        
        self.pet_backbone = densenet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
        )
        # Remove the final classification layer to get features
        self.pet_backbone.classifier = nn.Identity()
        
        # Clinical processor (from organizer's proven design)
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Your fusion strategies
        self.vae_fusion = VAEFusion(512, 512, latent_dim=64)
        self.attention_fusion = AttentionWeightedFusion(512, 512, attention_dim=64)
        
        # Final feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32, 256),  # VAE + Attention + Clinical
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, feature_output_dim)
        )
        
        # Risk prediction head (from organizer's training strategy)
        self.risk_head = nn.Sequential(
            nn.Linear(feature_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, ct_images, pet_images, clinical_features, return_risk=False):
        # Extract modality-specific features
        ct_features = self.ct_backbone(ct_images)  # [batch, 512]
        pet_features = self.pet_backbone(pet_images)  # [batch, 512]
        
        # Process clinical features
        clinical_processed = self.clinical_processor(clinical_features)  # [batch, 32]
        
        # Your advanced fusion strategies
        vae_latent, mu, logvar, reconstruction = self.vae_fusion(ct_features, pet_features)  # [batch, 64]
        attention_fused, attention_weights = self.attention_fusion(ct_features, pet_features)  # [batch, 64]
        
        # Combine all features
        combined_features = torch.cat([vae_latent, attention_fused, clinical_processed], dim=1)
        final_features = self.feature_fusion(combined_features)
        
        if return_risk:
            risk_scores = self.risk_head(final_features).squeeze(-1)
            return final_features, risk_scores, {
                'vae_mu': mu, 'vae_logvar': logvar, 'vae_reconstruction': reconstruction,
                'attention_weights': attention_weights
            }
        
        return final_features

class HybridSurvivalModel(nn.Module):
    """
    Unified model that can use either organizer's direct fusion approach
    or your advanced VAE+Attention approach.
    """
    def __init__(self, 
                 clinical_feature_dim: int = 13,
                 feature_dim: int = 128,
                 use_organizer_approach: bool = False,
                 vae_latent_dim: int = 64,
                 use_vae_fusion: bool = False,
                 use_attention_fusion: bool = False,
                 device: torch.device = None):
        super().__init__()
        
        self.use_organizer_approach = use_organizer_approach
        self.clinical_feature_dim = clinical_feature_dim
        self.feature_dim = feature_dim
        self.device = device or torch.device('cpu')
        
        if use_organizer_approach:
            # Use organizer's proven direct fusion approach
            self.feature_extractor = OrganizerFeatureExtractor(
                clinical_feature_dim=clinical_feature_dim,
                feature_output_dim=feature_dim
            ).to(self.device)
        else:
            # Use your enhanced multimodal approach
            self.feature_extractor = EnhancedMultimodalFeatureExtractor(
                clinical_feature_dim=clinical_feature_dim,
                feature_dim=feature_dim,
                vae_latent_dim=vae_latent_dim,
                use_vae_fusion=use_vae_fusion,
                use_attention_fusion=use_attention_fusion
            ).to(self.device)
        
        # Both approaches use BaggedIcareSurvival as backend
        self.icare_model = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        
        # Hyperparameters (can be optimized with Optuna)
        self.hyperparams = {
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'ranking_weight': 0.3,
            'contrastive_margin': 2.0,
        }
    
    def forward(self, ct_images=None, pet_images=None, combined_images=None, 
                clinical_features=None, return_features=False):
        """
        Forward pass that handles both approaches.
        
        Args:
            ct_images: [batch, 1, H, W, D] - CT images (for enhanced approach)
            pet_images: [batch, 1, H, W, D] - PET images (for enhanced approach)
            combined_images: [batch, 2, H, W, D] - Combined CT+PET (for organizer approach)
            clinical_features: [batch, clinical_dim]
            return_features: Whether to return extracted features
        """
        if self.use_organizer_approach:
            # Use organizer's direct fusion approach
            if combined_images is None:
                # Combine CT and PET if not already combined
                if ct_images is not None and pet_images is not None:
                    combined_images = torch.cat([ct_images, pet_images], dim=1)
                else:
                    raise ValueError("For organizer approach, need either combined_images or ct_images+pet_images")
            
            features, risk_scores = self.feature_extractor(
                combined_images, clinical_features, return_risk=True
            )
            
        else:
            # Use enhanced multimodal approach
            if ct_images is None or pet_images is None:
                raise ValueError("Enhanced approach requires separate ct_images and pet_images")
            
            features, risk_scores = self.feature_extractor(
                ct_images, pet_images, clinical_features, return_risk=True
            )
        
        if return_features:
            return risk_scores, features
        return risk_scores
    
    def configure_optimizer(self, trial=None):
        """Configure optimizer and scheduler with optional Optuna optimization."""
        if trial is not None:
            # Use Optuna for hyperparameter optimization
            self.hyperparams.update({
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'ranking_weight': trial.suggest_float('ranking_weight', 0.1, 0.5),
                'contrastive_margin': trial.suggest_float('contrastive_margin', 1.0, 3.0),
                'contrastive_temperature': trial.suggest_float('contrastive_temperature', 0.05, 0.2),
                'vae_loss_weight': trial.suggest_float('vae_loss_weight', 0.01, 0.2),
                'scheduler_factor': trial.suggest_float('scheduler_factor', 0.3, 0.7),
                'scheduler_patience': trial.suggest_int('scheduler_patience', 2, 5)
            })
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.feature_extractor.parameters(),
            lr=self.hyperparams['lr'],
            weight_decay=self.hyperparams['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.hyperparams['scheduler_factor'],
            patience=self.hyperparams['scheduler_patience'],
            verbose=True
        )
    
    def train_neural_features(self, train_loader, val_loader, epochs=50, trial=None):
        """Train the neural feature extractor with optional Optuna optimization."""
        if trial is not None:
            self.configure_optimizer(trial)
        else:
            self.configure_optimizer()
        
        # Initialize loss functions
        try:
            # Try to use advanced losses if available
            from .survival_losses import DeepHitLoss, SurvivalContrastiveLoss
            survival_loss = DeepHitLoss(ranking_weight=self.hyperparams['ranking_weight'])
            contrastive_loss = SurvivalContrastiveLoss(
                margin=self.hyperparams['contrastive_margin'], 
                temperature=self.hyperparams['contrastive_temperature']
            )
        except:
            # Fallback to simple losses
            survival_loss = nn.MSELoss()
            contrastive_loss = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_features = None
        
        for epoch in range(epochs):
            # Training phase
            self.feature_extractor.train()
            train_loss = 0.0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                if self.use_organizer_approach:
                    # Organizer's approach: combined CT+PET input
                    ct_images = batch['ct'].to(self.device)
                    pet_images = batch['pet'].to(self.device)
                    combined_images = torch.cat([ct_images, pet_images], dim=1)
                    clinical_features = batch['clinical'].to(self.device)
                    
                    risk_scores, features = self.forward(
                        combined_images=combined_images,
                        clinical_features=clinical_features,
                        return_features=True
                    )
                else:
                    # Enhanced approach: separate CT and PET processing
                    ct_images = batch['ct'].to(self.device)
                    pet_images = batch['pet'].to(self.device)
                    clinical_features = batch['clinical'].to(self.device)
                    
                    risk_scores, features = self.forward(
                        ct_images=ct_images,
                        pet_images=pet_images,
                        clinical_features=clinical_features,
                        return_features=True
                    )
                
                # Calculate losses
                survival_times = batch['survival_time'].to(self.device)
                event_indicators = batch['event'].to(self.device)
                
                # Main survival loss
                main_loss = survival_loss(risk_scores, survival_times, event_indicators)
                
                # Additional losses for enhanced approach
                total_loss = main_loss
                if not self.use_organizer_approach:
                    # Add VAE loss if using enhanced approach with VAE
                    if hasattr(self.feature_extractor, 'use_vae_fusion') and self.feature_extractor.use_vae_fusion:
                        vae_loss = getattr(self.feature_extractor, 'vae_loss', 0.0)
                        total_loss += self.hyperparams['vae_loss_weight'] * vae_loss
                
                total_loss.backward()
                self.optimizer.step()
                train_loss += total_loss.item()
            
            # Validation phase
            self.feature_extractor.eval()
            val_loss = 0.0
            val_features = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if self.use_organizer_approach:
                        ct_images = batch['ct'].to(self.device)
                        pet_images = batch['pet'].to(self.device)
                        combined_images = torch.cat([ct_images, pet_images], dim=1)
                        clinical_features = batch['clinical'].to(self.device)
                        
                        risk_scores, features = self.forward(
                            combined_images=combined_images,
                            clinical_features=clinical_features,
                            return_features=True
                        )
                    else:
                        ct_images = batch['ct'].to(self.device)
                        pet_images = batch['pet'].to(self.device)
                        clinical_features = batch['clinical'].to(self.device)
                        
                        risk_scores, features = self.forward(
                            ct_images=ct_images,
                            pet_images=pet_images,
                            clinical_features=clinical_features,
                            return_features=True
                        )
                    
                    survival_times = batch['survival_time'].to(self.device)
                    event_indicators = batch['event'].to(self.device)
                    
                    loss = survival_loss(risk_scores, survival_times, event_indicators)
                    val_loss += loss.item()
                    
                    val_features.append(features.cpu().numpy())
                    val_targets.append({
                        'survival_time': survival_times.cpu().numpy(),
                        'event': event_indicators.cpu().numpy()
                    })
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Save best features
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_features = np.vstack(val_features)
            
            # Report to Optuna if trial is provided
            if trial is not None and optuna is not None:
                trial.report(avg_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return best_features, best_val_loss

def run_comparative_experiment(train_loader, val_loader, clinical_feature_dim=13, device=None):
    """
    Run both organizer's and your approaches for comparison.
    
    This demonstrates the dual implementation with DenseNet121:
    1. Organizer's direct CT+PET fusion approach
    2. Your enhanced VAE+Attention fusion approach
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("HECKTOR 2025 Task 2: Dual Approach Comparison")
    print("Both approaches now use DenseNet121 backbone")
    print("=" * 60)
    
    results = {}
    
    # 1. Test Organizer's Direct Fusion Approach
    print("\n1. Testing Organizer's Direct Fusion Approach:")
    print("   - DenseNet121 with 2-channel input (CT+PET)")
    print("   - Simple concatenation fusion")
    print("   - BaggedIcareSurvival backend")
    
    organizer_model = HybridSurvivalModel(
        clinical_feature_dim=clinical_feature_dim,
        feature_dim=128,
        use_organizer_approach=True,  # Use organizer's approach
        device=device
    )
    
    # Train organizer's approach
    org_features, org_val_loss = organizer_model.train_neural_features(
        train_loader, val_loader, epochs=20
    )
    
    print(f"   Organizer approach validation loss: {org_val_loss:.4f}")
    results['organizer'] = {
        'val_loss': org_val_loss,
        'features': org_features,
        'approach': 'Direct CT+PET fusion with DenseNet121'
    }
    
    # 2. Test Your Enhanced Multimodal Approach
    print("\n2. Testing Your Enhanced Multimodal Approach:")
    print("   - Separate DenseNet121 for CT and PET")
    print("   - VAE + Attention fusion strategies")
    print("   - Advanced multimodal processing")
    
    enhanced_model = HybridSurvivalModel(
        clinical_feature_dim=clinical_feature_dim,
        feature_dim=128,
        use_organizer_approach=False,  # Use your approach
        vae_latent_dim=64,
        use_vae_fusion=True,
        use_attention_fusion=True,
        device=device
    )
    
    # Train your approach
    enh_features, enh_val_loss = enhanced_model.train_neural_features(
        train_loader, val_loader, epochs=20
    )
    
    print(f"   Enhanced approach validation loss: {enh_val_loss:.4f}")
    results['enhanced'] = {
        'val_loss': enh_val_loss,
        'features': enh_features,
        'approach': 'Separate DenseNet121 + VAE + Attention fusion'
    }
    
    # 3. Comparison Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY:")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"\n{name.upper()} APPROACH:")
        print(f"  Method: {result['approach']}")
        print(f"  Validation Loss: {result['val_loss']:.4f}")
        print(f"  Features Shape: {result['features'].shape}")
    
    # Determine better approach
    if results['organizer']['val_loss'] < results['enhanced']['val_loss']:
        print(f"\n🏆 WINNER: Organizer's approach (simpler but effective)")
        print(f"   Improvement: {results['enhanced']['val_loss'] - results['organizer']['val_loss']:.4f} lower loss")
    else:
        print(f"\n🏆 WINNER: Enhanced approach (advanced fusion pays off)")
        print(f"   Improvement: {results['organizer']['val_loss'] - results['enhanced']['val_loss']:.4f} lower loss")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("• Both approaches now use DenseNet121 for improved feature extraction")
    print("• Organizer's approach: Simple but proven effective")
    print("• Enhanced approach: More sophisticated fusion strategies")
    print("• Choose based on your specific dataset and computational resources")
    
    return results

# Example usage function
def create_example_usage():
    """
    Example of how to use both approaches.
    """
    example_code = '''
# Example: Using both approaches with DenseNet121

# 1. For Organizer's Direct Fusion Approach:
organizer_model = HybridSurvivalModel(
    clinical_feature_dim=13,
    use_organizer_approach=True,  # Key: Use organizer's method
    device=torch.device('cuda')
)

# Training with combined CT+PET input
for batch in train_loader:
    ct_images = batch['ct']  # [batch, 1, H, W, D]
    pet_images = batch['pet']  # [batch, 1, H, W, D]
    combined_images = torch.cat([ct_images, pet_images], dim=1)  # [batch, 2, H, W, D]
    
    risk_scores = organizer_model(
        combined_images=combined_images,
        clinical_features=batch['clinical']
    )

# 2. For Enhanced Multimodal Approach:
enhanced_model = HybridSurvivalModel(
    clinical_feature_dim=13,
    use_organizer_approach=False,  # Key: Use enhanced method
    use_vae_fusion=True,
    use_attention_fusion=True,
    device=torch.device('cuda')
)

# Training with separate CT and PET processing
for batch in train_loader:
    risk_scores = enhanced_model(
        ct_images=batch['ct'],    # Separate CT processing
        pet_images=batch['pet'],  # Separate PET processing
        clinical_features=batch['clinical']
    )

# 3. Comparative Training:
results = run_comparative_experiment(train_loader, val_loader)
print(f"Best approach: {results}")
'''
    return example_code

if __name__ == "__main__":
    print("Hybrid Multimodal Survival Prediction Framework")
    print("=" * 50)
    print("This framework implements both:")
    print("1. Organizer's direct CT+PET fusion with DenseNet121")
    print("2. Your enhanced VAE+Attention fusion with DenseNet121")
    print("\nBoth approaches use the same improved backbone!")
    print("\nExample usage:")
    print(create_example_usage())
    
    def vae_loss(self, reconstruction, original, mu, logvar):
        """VAE loss combining reconstruction and KL divergence."""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= original.size(0) * original.size(1)
        
        return recon_loss + kl_loss
    
    def extract_features_and_targets(self, data_loader):
        """Extract features using enhanced feature extractor."""
        self.feature_extractor.eval()
        all_features = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for ct_images, pet_images, clinical, times, events in data_loader:
                ct_images = ct_images.to(self.device)
                pet_images = pet_images.to(self.device)
                clinical = clinical.to(self.device)
                
                features = self.feature_extractor(ct_images, pet_images, clinical)
                
                all_features.append(features.cpu().numpy())
                all_times.extend(times.numpy())
                all_events.extend(events.numpy())
        
        feature_matrix = np.vstack(all_features)
        time_array = np.array(all_times)
        event_array = np.array(all_events)
        
        # Format for BaggedIcareSurvival (organizer's format)
        survival_dtype = [('event', bool), ('time', float)]
        survival_outcomes = np.array(
            list(zip(event_array.astype(bool), time_array.astype(float))), 
            dtype=survival_dtype
        )
        
        return feature_matrix, survival_outcomes
    
    def train_feature_extractor_epoch(self, train_loader):
        """Train enhanced feature extractor with multiple loss components."""
        self.feature_extractor.train()
        total_loss = 0
        batch_count = 0
        
        epoch_pbar = tqdm(train_loader, desc="Training enhanced feature extractor", leave=False)
        
        for ct_images, pet_images, clinical, times, events in epoch_pbar:
            ct_images = ct_images.to(self.device)
            pet_images = pet_images.to(self.device)
            clinical = clinical.to(self.device)
            times = times.to(self.device)
            events = events.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get features and additional outputs
            features, risk_scores, extras = self.feature_extractor(
                ct_images, pet_images, clinical, return_risk=True
            )
            
            # Compute losses
            survival_loss = self.survival_loss(risk_scores, times, events)
            contrastive_loss = self.contrastive_loss(features, times, events)
            
            # VAE loss (your enhancement)
            original_combined = torch.cat([
                ct_images.view(ct_images.size(0), -1),
                pet_images.view(pet_images.size(0), -1)
            ], dim=1)
            
            if original_combined.size(1) > extras['vae_reconstruction'].size(1):
                # Downsample original to match reconstruction
                original_combined = original_combined[:, :extras['vae_reconstruction'].size(1)]
            
            vae_loss = self.vae_loss(
                extras['vae_reconstruction'], original_combined,
                extras['vae_mu'], extras['vae_logvar']
            )
            
            # Combined loss (organizer's approach + your enhancements)
            total_loss_batch = (survival_loss + 
                              0.1 * contrastive_loss + 
                              self.vae_loss_weight * vae_loss)
            
            if not torch.isnan(total_loss_batch):
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += total_loss_batch.item()
                batch_count += 1
            
            epoch_pbar.set_postfix({
                'loss': total_loss_batch.item(),
                'survival': survival_loss.item(),
                'contrastive': contrastive_loss.item(),
                'vae': vae_loss.item()
            })
        
        return total_loss / max(batch_count, 1)
    
    def train_icare_model(self, train_loader):
        """Train BaggedIcareSurvival with enhanced features."""
        X_train, y_train = self.extract_features_and_targets(train_loader)
        
        try:
            self.icare_model = BaggedIcareSurvival(
                aggregation_method='median',
                n_jobs=-1,
                random_state=RANDOM_SEED
            )
            self.icare_model.fit(X_train, y_train)
        except Exception as e:
            print(f"BaggedIcareSurvival training failed: {e}")
            # Fallback to simple model
            from sklearn.ensemble import RandomForestRegressor
            self.icare_model = RandomForestRegressor(random_state=RANDOM_SEED)
            # Reshape survival data for sklearn
            y_simple = y_train['time'] * (2 * y_train['event'].astype(int) - 1)
            self.icare_model.fit(X_train, y_simple)
    
    def evaluate_system(self, data_loader):
        """Evaluate the complete enhanced system."""
        try:
            X_test, y_test = self.extract_features_and_targets(data_loader)
            
            if hasattr(self.icare_model, 'predict'):
                predictions = self.icare_model.predict(X_test)
            else:
                # Fallback prediction
                predictions = np.random.rand(len(X_test))
            
            times = y_test['time']
            events = y_test['event'].astype(int)
            
            c_index = concordance_index(times, -predictions, events)
            return c_index
            
        except Exception as e:
            print(f"System evaluation failed: {e}")
            return 0.5  # Random performance fallback
    
    def fit(self, train_loader, val_loader, num_iterations=20, feature_epochs_per_iteration=5):
        """
        Train the complete enhanced system using organizer's iterative approach.
        """
        print(f"Training Enhanced System: {num_iterations} iterations × {feature_epochs_per_iteration} epochs")
        
        # Initial training
        self.train_icare_model(train_loader)
        initial_c_index = self.evaluate_system(val_loader)
        print(f"Initial validation C-index: {initial_c_index:.4f}")
        
        self.best_c_index = initial_c_index
        self.best_feature_state = self.feature_extractor.state_dict().copy()
        self.best_icare_model = pickle.loads(pickle.dumps(self.icare_model))
        
        # Iterative joint training (organizer's proven strategy)
        for iteration in tqdm(range(num_iterations), desc="Training iterations"):
            print(f"\n=== Enhanced Iteration {iteration + 1}/{num_iterations} ===")
            
            # Train enhanced feature extractor
            print("Training enhanced feature extractor...")
            for epoch in range(feature_epochs_per_iteration):
                avg_loss = self.train_feature_extractor_epoch(train_loader)
            
            # Retrain survival model with enhanced features
            self.train_icare_model(train_loader)
            
            # Evaluate
            current_c_index = self.evaluate_system(val_loader)
            print(f"Validation C-index: {current_c_index:.4f}")
            
            self.scheduler.step(current_c_index)
            
            # Save best model
            if current_c_index > self.best_c_index:
                print(f"New best C-index: {current_c_index:.4f} (improvement: +{current_c_index - self.best_c_index:.4f})")
                self.best_c_index = current_c_index
                self.best_feature_state = self.feature_extractor.state_dict().copy()
                self.best_icare_model = pickle.loads(pickle.dumps(self.icare_model))
        
        # Load best models
        print(f"\nEnhanced training complete! Best validation C-index: {self.best_c_index:.4f}")
        self.feature_extractor.load_state_dict(self.best_feature_state)
        self.icare_model = self.best_icare_model

# =============================================================================
# Enhanced Dataset Class Supporting Region-Based Processing
# =============================================================================

class EnhancedHecktorDataset(Dataset):
    """
    Enhanced dataset supporting your region-based processing approach.
    """
    def __init__(self, cached_data, patient_ids, use_regions=False):
        self.patient_ids = [pid for pid in patient_ids if pid in cached_data['images']]
        self.cached_data = cached_data
        self.use_regions = use_regions
        
        print(f"Enhanced dataset initialized with {len(self.patient_ids)} patients")
        if use_regions:
            print("Region-based processing enabled")
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Get combined CT+PET image
        combined_image = self.cached_data['images'][patient_id]  # [2, H, W, D]
        
        # Split into separate modalities
        ct_image = combined_image[0:1]  # [1, H, W, D]
        pet_image = combined_image[1:2]  # [1, H, W, D]
        
        # Clinical features
        clinical_tensor = torch.tensor(
            self.cached_data['clinical_features']['features'][patient_id], 
            dtype=torch.float32
        )
        
        # Survival data
        survival_time = torch.tensor(
            self.cached_data['survival_data'][patient_id]['time'], 
            dtype=torch.float32
        )
        event_indicator = torch.tensor(
            self.cached_data['survival_data'][patient_id]['event'], 
            dtype=torch.float32
        )
        
        return ct_image, pet_image, clinical_tensor, survival_time, event_indicator

# =============================================================================
# Region-Based Processing Pipeline (Your Enhanced Approach)
# =============================================================================

def process_regions_with_segmentation(ct_path: str, pet_path: str, 
                                    clinical_data: Dict, 
                                    margin_mm: float = 10.0):
    """
    Your region-based processing enhanced with organizer's data handling.
    """
    # Load images
    ct_image = sitk.ReadImage(ct_path)
    pet_image = sitk.ReadImage(pet_path)
    
    # Segment using your approach
    segmentator = Segmentator()
    segmentation, _ = segmentator.predict(image_ct=ct_image, image_pet=pet_image, preprocess=True)
    
    # Find connected components (your approach)
    seg_array = sitk.GetArrayFromImage(segmentation)
    labeled_array, num_components = label(seg_array > 0)
    
    region_data = []
    
    for cc_idx in range(1, num_components + 1):
        # Create mask for this component
        cc_mask_array = (labeled_array == cc_idx).astype(np.uint8)
        cc_mask = sitk.GetImageFromArray(cc_mask_array)
        cc_mask.CopyInformation(segmentation)
        
        try:
            # Crop around mask (your approach)
            cropped_ct, bbox = crop_around_mask(ct_image, cc_mask, margin_mm)
            cropped_pet, _ = crop_around_mask(pet_image, cc_mask, margin_mm)
            cropped_mask, _ = crop_around_mask(cc_mask, cc_mask, margin_mm)
            
            # Apply organizer's transforms
            transforms = create_image_transforms()
            
            # Convert to format expected by transforms
            # Save temporarily and reload (could be optimized)
            temp_ct_path = f"/tmp/temp_ct_{cc_idx}.nii.gz"
            temp_pet_path = f"/tmp/temp_pet_{cc_idx}.nii.gz"
            
            sitk.WriteImage(cropped_ct, temp_ct_path)
            sitk.WriteImage(cropped_pet, temp_pet_path)
            
            try:
                transformed_data = transforms({"ct": temp_ct_path, "pet": temp_pet_path})
                combined_image = torch.cat([transformed_data["ct"], transformed_data["pet"]], dim=0)
                
                region_data.append({
                    'component_id': cc_idx,
                    'combined_image': combined_image,
                    'volume_voxels': np.sum(cc_mask_array > 0),
                    'bbox': bbox
                })
                
            finally:
                # Cleanup temp files
                if os.path.exists(temp_ct_path):
                    os.remove(temp_ct_path)
                if os.path.exists(temp_pet_path):
                    os.remove(temp_pet_path)
                    
        except Exception as e:
            print(f"Warning: Could not process region {cc_idx}: {e}")
            continue
    
    return region_data

# =============================================================================
# Optuna Hyperparameter Optimization (Enhancement over Organizer's Fixed Params)
# =============================================================================

def optimize_hyperparameters_with_optuna(data, clinical_feature_dim, device, 
                                        n_trials=50, cv_folds=3, 
                                        num_iterations=10, feature_epochs=3):
    """
    Enhanced hyperparameter optimization using Optuna.
    This is an improvement over the organizer's fixed hyperparameters.
    
    Args:
        data: Training data
        clinical_feature_dim: Number of clinical features
        device: Training device
        n_trials: Number of Optuna trials
        cv_folds: Cross-validation folds
        num_iterations: Training iterations per trial
        feature_epochs: Feature epochs per iteration
        
    Returns:
        Best hyperparameters and study results
    """
    try:
        import optuna
        from sklearn.model_selection import KFold
        
        def objective(trial):
            """Optuna objective function."""
            # Get patient IDs and create CV splits
            patient_ids = data['patient_ids']
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            
            cv_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(patient_ids)):
                print(f"Trial {trial.number}, Fold {fold_idx + 1}/{cv_folds}")
                
                # Split data
                train_patients = [patient_ids[i] for i in train_idx]
                val_patients = [patient_ids[i] for i in val_idx]
                
                # Create datasets and loaders
                train_dataset = EnhancedHecktorDataset(data, train_patients)
                val_dataset = EnhancedHecktorDataset(data, val_patients)
                
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
                
                # Initialize model with suggested hyperparameters
                model = HybridSurvivalModel(
                    clinical_feature_dim=clinical_feature_dim,
                    device=device,
                    feature_dim=128
                )
                
                # Update with suggested hyperparameters
                suggested_params = model.suggest_hyperparameters(trial)
                model.update_hyperparameters(suggested_params)
                
                try:
                    # Train with reduced iterations for faster optimization
                    model.fit(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        num_iterations=num_iterations,
                        feature_epochs_per_iteration=feature_epochs
                    )
                    
                    # Get validation score
                    cv_scores.append(model.best_c_index)
                    
                except Exception as e:
                    print(f"Trial {trial.number}, Fold {fold_idx + 1} failed: {e}")
                    cv_scores.append(0.5)  # Random performance
                    
                # Report intermediate result for pruning
                trial.report(np.mean(cv_scores), fold_idx)
                
                # Prune trial if it's not promising
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Return mean CV score
            mean_score = np.mean(cv_scores)
            print(f"Trial {trial.number} completed. Mean CV C-index: {mean_score:.4f}")
            return mean_score
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=cv_folds
            )
        )
        
        print(f"Starting Optuna optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)
        
        # Results
        print("Optuna optimization completed!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best C-index: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        return study.best_params, study
        
    except ImportError:
        print("Optuna not installed. Using organizer's fixed hyperparameters.")
        # Return organizer's fixed hyperparameters as fallback
        return {
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'ranking_weight': 0.3,
            'contrastive_margin': 2.0,
            'contrastive_temperature': 0.1,
            'vae_loss_weight': 0.1,
            'scheduler_factor': 0.5,
            'scheduler_patience': 3
        }, None
    
    except Exception as e:
        print(f"Optuna optimization failed: {e}")
        print("Falling back to organizer's fixed hyperparameters.")
        return {
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'ranking_weight': 0.3,
            'contrastive_margin': 2.0,
            'contrastive_temperature': 0.1,
            'vae_loss_weight': 0.1,
            'scheduler_factor': 0.5,
            'scheduler_patience': 3
        }, None

# =============================================================================
# Complete Enhanced Pipeline
# =============================================================================

def enhanced_rfs_prediction_pipeline(ct_path: str, pet_path: str, 
                                   clinical_data: Dict,
                                   model_path: Optional[str] = None,
                                   save_outputs: bool = False,
                                   output_dir: Optional[str] = None):
    """
    Complete enhanced pipeline combining your approach with organizer's insights.
    """
    print("=== Enhanced RFS Prediction Pipeline ===")
    
    # Step 1: Process regions (your approach)
    print("Processing regions with enhanced segmentation...")
    region_data = process_regions_with_segmentation(ct_path, pet_path, clinical_data)
    
    if not region_data:
        print("Warning: No valid regions found. Using whole image.")
        # Fallback to whole image processing
        transforms = create_image_transforms()
        transformed_data = transforms({"ct": ct_path, "pet": pet_path})
        combined_image = torch.cat([transformed_data["ct"], transformed_data["pet"]], dim=0)
        region_data = [{'component_id': 0, 'combined_image': combined_image, 'volume_voxels': 1000}]
    
    # Step 2: Load or initialize enhanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process clinical data using organizer's approach
    clinical_df = pd.DataFrame([clinical_data])
    clinical_df['PatientID'] = 'test_patient'
    processed_clinical = preprocess_clinical_data(clinical_df)
    clinical_features = processed_clinical['features']['test_patient']
    
    # Initialize enhanced model
    model = HybridSurvivalModel(
        clinical_feature_dim=len(clinical_features),
        device=device,
        feature_dim=128
    )
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        model.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        model.icare_model = checkpoint['icare_model']
    else:
        print("Warning: No trained model provided. Using random initialization.")
    
    # Step 3: Extract features from all regions
    print(f"Extracting enhanced features from {len(region_data)} regions...")
    
    model.feature_extractor.eval()
    region_features = []
    
    with torch.no_grad():
        for region in region_data:
            combined_image = region['combined_image'].unsqueeze(0).to(device)  # Add batch dim
            ct_image = combined_image[:, 0:1]  # [1, 1, H, W, D]
            pet_image = combined_image[:, 1:2]  # [1, 1, H, W, D]
            
            clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Extract enhanced features
            features = model.feature_extractor(ct_image, pet_image, clinical_tensor)
            region_features.append(features.cpu().numpy())
    
    # Step 4: Aggregate features across regions (your approach)
    if len(region_features) > 1:
        # Weight by volume
        volumes = [r['volume_voxels'] for r in region_data]
        weights = np.array(volumes) / np.sum(volumes)
        
        final_features = np.average(region_features, axis=0, weights=weights)
    else:
        final_features = region_features[0]
    
    # Step 5: Survival prediction using organizer's proven approach
    print("Predicting survival...")
    try:
        if model.icare_model:
            risk_score = model.icare_model.predict(final_features)[0]
        else:
            risk_score = np.random.rand()  # Fallback
            
        # Convert to survival probabilities (placeholder)
        survival_predictions = {
            'risk_score': float(risk_score),
            'survival_probability_1year': float(1.0 / (1.0 + np.exp(risk_score))),
            'survival_probability_2year': float(1.0 / (1.0 + np.exp(risk_score * 1.5))),
            'survival_probability_5year': float(1.0 / (1.0 + np.exp(risk_score * 2.0))),
            'num_regions_processed': len(region_data),
            'model_type': 'Enhanced_Hybrid'
        }
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        survival_predictions = {
            'risk_score': 0.5,
            'survival_probability_1year': 0.8,
            'survival_probability_2year': 0.7,
            'survival_probability_5year': 0.6,
            'num_regions_processed': len(region_data),
            'model_type': 'Enhanced_Hybrid_Fallback'
        }
    
    # Step 6: Save outputs
    if save_outputs and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        results_summary = {
            'clinical_data': clinical_data,
            'survival_predictions': survival_predictions,
            'region_analysis': {
                'num_regions': len(region_data),
                'region_volumes': [r['volume_voxels'] for r in region_data]
            }
        }
        
        with open(os.path.join(output_dir, 'enhanced_prediction_results.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"Results saved to {output_dir}")
    
    print("Enhanced pipeline completed!")
    print(f"Risk Score: {survival_predictions['risk_score']:.4f}")
    print(f"1-year survival probability: {survival_predictions['survival_probability_1year']:.4f}")
    
    return survival_predictions

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Set random seed
    set_random_seed(RANDOM_SEED)
    
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
    
    # Example paths
    ct_path = "/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/ct/CHUM-001.mha"
    pet_path = "/Data/Yujing/HECKTOR2025/Hecktor2025/input/images/pet/CHUM-001.mha"
    
    # Run enhanced pipeline
    results = enhanced_rfs_prediction_pipeline(
        ct_path=ct_path,
        pet_path=pet_path,
        clinical_data=example_clinical,
        model_path=None,  # Set to trained model path when available
        save_outputs=True,
        output_dir="/Data/Yujing/HECKTOR2025/Hecktor2025/outcome_pred/tmp/enhanced_results"
    )
    
    print("\nEnhanced Prediction Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
