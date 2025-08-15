#!/usr/bin/env python3
"""
Generate predictions for all 1347 cases using 5-fold cross-validation models
Fixed version based on training script insights
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add mSTAR path
mstar_path = os.path.join(os.getcwd(), 'mSTAR', 'downstream_task', 'multimodal_survival')
sys.path.insert(0, mstar_path)

from models.Multimodal.Porpoise.network import Porpoise
from models.Multimodal.Porpoise.engine import Engine


class PredictionGenerator1347Fixed:
    """Generate predictions for all 1347 cases using 5-fold CV models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Paths
        self.dataset_file = 'scc_survival_1347.csv'  # Fixed: use actual filename
        self.model_results_dir = 'scc_results_full_1347/Porpoise-[scc_1347]'
        self.output_dir = 'correct_predictions'
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset
        self.df = pd.read_csv(self.dataset_file)
        print(f"📊 Loaded dataset: {len(self.df)} cases")
        
        # Check fold distribution
        fold_counts = self.df['split'].value_counts().sort_index()
        print("📁 Fold distribution:")
        for fold, count in fold_counts.items():
            print(f"   {fold}: {count} cases")
    
    def find_model_checkpoint(self, fold):
        """Find the best model checkpoint for a specific fold"""
        fold_dir = os.path.join(self.model_results_dir, str(fold))
        if not os.path.exists(fold_dir):
            print(f"❌ Fold directory not found: {fold_dir}")
            return None
        
        # Use the exact checkpoint file names provided
        checkpoint_names = {
            0: "model_best_0.6756_8.pth.tar",
            1: "model_best_0.7024_9.pth.tar", 
            2: "model_best_0.7007_11.pth.tar",
            3: "model_best_0.7224_8.pth.tar",
            4: "model_best_0.6795_9.pth.tar"
        }
        
        checkpoint_name = checkpoint_names.get(fold)
        if checkpoint_name is None:
            print(f"❌ Checkpoint name not found for fold {fold}")
            return None
        
        checkpoint_path = os.path.join(fold_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return None
        
        print(f"📁 Found checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_model(self, checkpoint_path, fold):
        """Load a trained model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create model (same architecture as training)
            model = Porpoise(
                omic_input_dim=5768,  # RNA (5000) + Text (768) - same as training
                path_input_dim=1024,  # WSI features
                fusion='lrb',         # Low-rank bilinear fusion (from training)
                dropout=0.4,          # From training config
                dropinput=0.2,        # From training config
                n_classes=4           # 4 hazard levels (0, 1, 2, 3)
            )
            
            # Load state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            print(f"✅ Model loaded successfully for fold {fold}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model for fold {fold}: {e}")
            return None
    

    
    def process_single_case(self, case_id, wsi_path, rna_path, model):
        """Process a single case and generate prediction"""
        try:
            # Load WSI features - simplified approach like 1211 script
            if os.path.exists(wsi_path):
                wsi_features = torch.load(wsi_path, map_location='cpu').float()
                if isinstance(wsi_features, dict):
                    wsi_features = wsi_features['features']
                wsi_features = wsi_features.unsqueeze(0)  # Add batch dimension
                print(f"✅ WSI features processed: {wsi_features.shape}")
            else:
                wsi_features = torch.zeros(1, 1, 1024) # Placeholder if WSI not found
                print(f"⚠️  WSI file not found: {wsi_path}. Using placeholder.")
            
            # Load RNA features - simplified approach like 1211 script
            if os.path.exists(rna_path):
                rna_df = pd.read_csv(rna_path, sep='\t')
                gene_names = rna_df['Gene'].tolist()
                indices = torch.tensor(rna_df['Index'].values, dtype=torch.long)
                values = torch.tensor(rna_df['Value'].values, dtype=torch.float32)
                rna_values = values
            else:
                print(f"⚠️  RNA features not found: {rna_path}")
                return None
            
            # Load text features (try to load actual features, fallback to zeros)
            text_path = wsi_path.replace('features', 'processed_pathology').replace('.pt', '_text.pt')
            if os.path.exists(text_path):
                try:
                    text_features = torch.load(text_path, map_location='cpu')
                    if isinstance(text_features, torch.Tensor):
                        text_tensor = text_features
                    else:
                        text_tensor = torch.zeros(768, dtype=torch.float32)
                except:
                    text_tensor = torch.zeros(768, dtype=torch.float32)
            else:
                text_tensor = torch.zeros(768, dtype=torch.float32)
            
            # Combine RNA + Text (like in training)
            text_weight = 0.3
            enhanced_values = torch.cat([rna_values, text_weight * text_tensor])
            
            # Ensure tensors have correct dimensions and move to device
            if wsi_features.dim() == 2:
                wsi_features = wsi_features.unsqueeze(0)
            
            if enhanced_values.dim() == 1:
                enhanced_values = enhanced_values.unsqueeze(0)
            
            # Move tensors to device
            wsi_features = wsi_features.to(self.device)
            enhanced_values = enhanced_values.to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                hazards, survival_curves = model(x_path=wsi_features, x_omic=enhanced_values)
                
                # Convert to numpy
                hazards_np = hazards.cpu().numpy()
                survival_np = survival_curves.cpu().numpy()
                
                # Calculate risk score (mSTAR method - negative sum of survival probabilities)
                risk_score = -torch.sum(survival_curves, dim=1).item()
                
                print(f"✅ Prediction generated - Hazards: {hazards_np.shape}, Survival: {survival_np.shape}")
                
                return {
                    'case_id': case_id,
                    'hazards': hazards_np.flatten().tolist(),
                    'survival_curves': survival_np.flatten().tolist(),
                    'risk_score': risk_score
                }
                
        except Exception as e:
            print(f"❌ Error processing case {case_id}: {e}")
            return None
    
    def generate_predictions_for_fold(self, fold):
        """Generate predictions for all cases in a specific fold"""
        print(f"\n🔄 Generating predictions for fold {fold}")
        
        # Load model for this fold
        checkpoint_path = self.find_model_checkpoint(fold)
        if checkpoint_path is None:
            print(f"❌ Could not find checkpoint for fold {fold}")
            return []
        
        model = self.load_model(checkpoint_path, fold)
        if model is None:
            print(f"❌ Could not load model for fold {fold}")
            return []
        
        # Get cases for this fold
        fold_cases = self.df[self.df['split'] == f'fold_{fold}']
        print(f"📊 Processing {len(fold_cases)} cases for fold {fold}")
        
        predictions = []
        
        for idx, row in tqdm(fold_cases.iterrows(), total=len(fold_cases), desc=f"Fold {fold}"):
            case_id = row['ID']
            wsi_path = row['WSI']
            rna_path = row['RNA']
            
            # Process case
            prediction = self.process_single_case(case_id, wsi_path, rna_path, model)
            if prediction:
                # Add actual survival data and fold information
                prediction['actual_time'] = row['Event']
                prediction['actual_status'] = row['Status']
                prediction['fold'] = fold
                
                predictions.append(prediction)
        
        print(f"✅ Fold {fold}: Generated {len(predictions)} predictions")
        return predictions
    
    def generate_all_predictions(self):
        """Generate predictions for all 1347 cases using 5-fold CV"""
        print("🚀 Starting prediction generation for all 1347 cases...")
        
        all_predictions = []
        
        # Process each fold
        for fold in range(5):
            fold_predictions = self.generate_predictions_for_fold(fold)
            all_predictions.extend(fold_predictions)
        
        # Convert to DataFrame and save results
        if all_predictions:
            df = pd.DataFrame(all_predictions)
            
            # Save predictions as CSV
            output_file = os.path.join(self.output_dir, 'predictions_3modal_1347.csv')
            df.to_csv(output_file, index=False)
            
            print(f"\n🎉 Prediction generation complete!")
            print(f"📊 Total predictions: {len(df)}")
            print(f"📁 Results saved to: {output_file}")
            
            # Print summary statistics
            print(f"\n📊 Risk Score Statistics:")
            print(f"Mean: {df['risk_score'].mean():.3f}")
            print(f"Std: {df['risk_score'].std():.3f}")
            print(f"Min: {df['risk_score'].min():.3f}")
            print(f"Max: {df['risk_score'].max():.3f}")
            
            return df
        else:
            print("❌ No predictions generated")
            return None


def main():
    """Main function"""
    print("🎯 SCC 3-Modal Survival Analysis - Generate Predictions (1347 cases)")
    print("=" * 70)
    
    # Initialize generator
    generator = PredictionGenerator1347Fixed()
    
    # Generate predictions
    predictions_df = generator.generate_all_predictions()
    
    if predictions_df is not None:
        print(f"\n✅ Script completed successfully!")
        print(f"📊 Generated predictions for {len(predictions_df)} cases")
    else:
        print(f"\n❌ Script failed to generate predictions")


if __name__ == "__main__":
    main()
