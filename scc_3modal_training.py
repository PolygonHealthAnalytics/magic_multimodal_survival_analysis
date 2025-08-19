#!/usr/bin/env python3
"""
SCC 3-Modal Survival Analysis - CLEAN VERSION.
Uses pre-extracted text features from disk (run extract_text_features.py first).
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import warnings
warnings.filterwarnings('ignore')

# Add mSTAR path
mstar_path = os.path.join(os.getcwd(), 'mSTAR', 'downstream_task', 'multimodal_survival')
sys.path.insert(0, mstar_path)

from models.Multimodal.Porpoise.engine import Engine
from models.Multimodal.Porpoise.network import Porpoise
from utils.loss import NLLSurvLoss
from utils.util import set_seed
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler


class OptimizedMultimodalDataset(Dataset):
    """Optimized dataset with pre-saved text features"""
    
    def __init__(self, csv_file, data_root='data/patients', use_text_features=True):
        self.df = pd.read_csv(csv_file)
        self.data_root = data_root
        self.use_text_features = use_text_features
        
        print(f"[Dataset] Loaded {len(self.df)} cases from {csv_file}")
        
        # Check text feature availability
        if use_text_features:
            self._check_text_features()
        
        # Display dataset info
        alive_count = (self.df['Status'] == 0).sum()
        dead_count = (self.df['Status'] == 1).sum()
        print(f"[Dataset] Cases: {alive_count} Alive, {dead_count} Dead")
        
        # Create discrete labels (quartiles for stratification)
        self.df['Label'] = pd.qcut(self.df['Event'], q=4, labels=[0, 1, 2, 3])
    
    def _check_text_features(self):
        """Check availability of pre-extracted text features"""
        print("🔍 Checking pre-extracted text features...")
        
        available_count = 0
        missing_count = 0
        
        # Check sample of cases
        sample_cases = self.df.head(10)['ID'].tolist()
        
        for case_id in sample_cases:
            feature_file = os.path.join(self.data_root, case_id, 'processed_text_features', 'pubmedbert_features.npy')
            if os.path.exists(feature_file):
                available_count += 1
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"⚠️  Warning: {missing_count}/{len(sample_cases)} sample cases missing text features")
            print(f"💡 Run: python extract_text_features.py")
            print(f"🔄 Falling back to zero text features for missing cases")
        else:
            print(f"✅ Text features available for all sample cases")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row['ID']
        
        # Load WSI features
        wsi_path = row['WSI']
        if pd.notna(wsi_path) and os.path.exists(wsi_path):
            wsi_tensor = torch.load(wsi_path, weights_only=True).float()
        else:
            wsi_tensor = torch.zeros((1, 1024)).float()  # Fallback
        
        # Load RNA features
        rna_path = row['RNA']
        if pd.notna(rna_path) and os.path.exists(rna_path):
            rna_df = pd.read_csv(rna_path, sep='\t')
            gene_names = rna_df['Gene'].tolist()
            indices = torch.tensor(rna_df['Index'].values, dtype=torch.long)
            values = torch.tensor(rna_df['Value'].values, dtype=torch.float32)
        else:
            # Fallback RNA data
            gene_names = [f"gene_{i}" for i in range(5000)]
            indices = torch.arange(5000, dtype=torch.long)
            values = torch.zeros(5000, dtype=torch.float32)
        
        # Load pre-extracted text features from disk
        if self.use_text_features:
            feature_file = os.path.join(self.data_root, case_id, 'processed_text_features', 'pubmedbert_features.npy')
            if os.path.exists(feature_file):
                try:
                    text_features = np.load(feature_file)
                except:
                    text_features = np.zeros(768)
            else:
                text_features = np.zeros(768)
        else:
            text_features = np.zeros(768)
        
        text_tensor = torch.tensor(text_features, dtype=torch.float32)
        
        # Combine RNA (5000) + Text (768) = 5768 dimensional features
        # Apply text weight to reduce potential noise
        text_weight = 0.3  # Reduce text influence
        enhanced_values = torch.cat([values, text_weight * text_tensor])
        
        # Update gene names to include text features
        text_feature_names = [f"text_feature_{i}" for i in range(768)]
        enhanced_gene_names = gene_names + text_feature_names
        
        # Update indices
        enhanced_indices = torch.arange(len(enhanced_values), dtype=torch.long)
        
        rna_data = (enhanced_gene_names, enhanced_indices, enhanced_values)
        
        # Survival data
        event_time = row['Event']
        censorship = row['Status']
        label = row['Label']
        
        return case_id, wsi_tensor, rna_data, event_time, censorship, label
    
    def get_split(self, fold=0):
        """Get train/val splits for specified fold"""
        train_idx = self.df[self.df['split'] != f'fold_{fold}'].index.tolist()
        val_idx = self.df[self.df['split'] == f'fold_{fold}'].index.tolist()
        return train_idx, val_idx


class TrainingConfig:
    """Training configuration for optimized 3-modal analysis"""
    def __init__(self):
        # Data
        self.csv_file = 'scc_survival_full_1347.csv'
        self.data_root = 'data/patients'
        
        # Training
        self.num_epoch = 40  # Increased from 25
        self.batch_size = 1  # Required for MIL
        self.lr = 1e-4
        self.weight_decay = 5e-4
        self.seed = 42
        
        # Model (adjusted for enhanced features)
        self.model = 'Porpoise'
        self.fusion = 'lrb'
        self.dropout = 0.4
        self.dropinput = 0.2
        
        # Optimization
        self.optimizer = 'Adam'
        self.scheduler = 'cosine'
        self.loss = 'nll_surv'
        
        # Other
        self.tqdm = False
        self.log_data = True
        self.resume = None
        self.evaluate = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model_and_optimizer(config):
    """Create enhanced model with text features"""
    
    # Model with enhanced RNA dimension (5000 + 768 = 5768)
    model = Porpoise(
        omic_input_dim=5768,  # RNA (5000) + Text (768)
        path_input_dim=1024,  # WSI features
        fusion=config.fusion,
        dropout=config.dropout,
        dropinput=config.dropinput,
        n_classes=4
    )
    model = model.to(config.device)
    
    # Loss function
    loss_fn = NLLSurvLoss()
    
    # Optimizer
    optimizer = define_optimizer(config, model)
    
    # Scheduler
    scheduler = define_scheduler(config, optimizer)
    
    return model, loss_fn, optimizer, scheduler


def train_fold(config, dataset, fold):
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"🔄 FOLD {fold} - FULL DATASET 3-MODAL")
    print(f"{'='*60}")
    
    # Get splits
    train_idx, val_idx = dataset.get_split(fold)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Results directory
    results_dir = f"scc_results_full_1347/Porpoise-[scc_1347]/{fold}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Model and training components
    model, loss_fn, optimizer, scheduler = create_model_and_optimizer(config)
    
    # Engine
    engine = Engine(config, results_dir)
    
    # Train
    print(f"🚀 Fold {fold} - Starting full dataset training ({config.num_epoch} epochs)...")
    best_scores, best_epoch = engine.learning(model, train_loader, val_loader, loss_fn, optimizer, scheduler)
    
    print(f"✅ Fold {fold} Complete!")
    print(f"   Best C-index: {best_scores['mean']:.4f} at epoch {best_epoch}")
    
    return best_scores, best_epoch


def main():
    """Main training function"""
    print("🎯 SCC 3-Modal Survival Analysis - FULL DATASET (1347 cases)")
    print("=" * 60)
    
    # Setup
    config = TrainingConfig()
    set_seed(config.seed)
    
    print(f"📱 Device: {config.device}")
    print(f"📄 Dataset: {config.csv_file}")
    print(f"🧬 Enhanced Features: RNA (5000) + Text (768) = 5768 dims")
    print(f"⚡ Optimization: Pre-extracted text features from disk")
    print(f"📊 Full Dataset: 1347 cases (no outlier filtering)")
    
    # Load dataset with pre-saved text features
    dataset = OptimizedMultimodalDataset(config.csv_file, config.data_root, use_text_features=True)
    
    # 5-fold cross-validation
    fold_results = []
    
    for fold in range(5):
        try:
            best_scores, best_epoch = train_fold(config, dataset, fold)
            fold_results.append({
                'fold': fold,
                'c_index': best_scores['mean'],
                'best_epoch': best_epoch
            })
        except Exception as e:
            print(f"❌ Fold {fold} failed: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS - FULL DATASET 3-MODAL (1347 cases)")
    print(f"{'='*60}")
    
    if fold_results:
        c_indices = [r['c_index'] for r in fold_results]
        mean_c_index = np.mean(c_indices)
        std_c_index = np.std(c_indices)
        
        print(f"🎯 Cross-Validation Results:")
        for result in fold_results:
            print(f"   Fold {result['fold']}: C-index = {result['c_index']:.4f} (epoch {result['best_epoch']})")
        
        print(f"\n🏆 Overall Performance:")
        print(f"   Mean C-index: {mean_c_index:.4f} ± {std_c_index:.4f}")
        
        # Save results
        os.makedirs('scc_results_full_1347', exist_ok=True)
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv('scc_results_full_1347/final_results.csv', index=False)
        print(f"📁 Results saved to: scc_results_full_1347/final_results.csv")
    else:
        print("❌ No successful folds completed")


if __name__ == "__main__":
    main()