#!/usr/bin/env python3
"""
Text Feature Extraction Script
Pre-extracts PubMedBERT features for all cases and saves them to disk
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class TextFeatureExtractor:
    """Extract and save text features using PubMedBERT"""
    
    def __init__(self):
        print("🔄 Loading PubMedBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        self.model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"✅ PubMedBERT loaded on {self.device}")
    
    def extract_features(self, text):
        """Extract 768-dim features from text"""
        try:
            inputs = self.tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                text_features = outputs.last_hidden_state[:, 0, :].squeeze()  # (768,)
                
            return text_features.cpu().numpy()
            
        except Exception as e:
            print(f"   ❌ Error extracting features: {e}")
            return np.zeros(768)
    
    def process_case(self, case_id, data_root):
        """Process a single case and save features"""
        # Check if already processed
        output_path = os.path.join(data_root, case_id, 'processed_text_features')
        feature_file = os.path.join(output_path, 'pubmedbert_features.npy')
        
        if os.path.exists(feature_file):
            return "✅ Already processed"
        
        # Load pathology text
        text_path = os.path.join(data_root, case_id, 'processed_pathology')
        if not os.path.exists(text_path):
            return "❌ No pathology directory"
        
        text_files = [f for f in os.listdir(text_path) if f.endswith('.txt')]
        if not text_files:
            return "❌ No text files"
        
        try:
            text_file = os.path.join(text_path, text_files[0])
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                return "❌ Empty text"
            
            # Extract features
            features = self.extract_features(text)
            
            # Save features
            os.makedirs(output_path, exist_ok=True)
            np.save(feature_file, features)
            
            # Save metadata
            metadata = {
                'case_id': case_id,
                'text_length': len(text),
                'feature_shape': features.shape,
                'source_file': text_files[0]
            }
            
            metadata_file = os.path.join(output_path, 'metadata.json')
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return f"✅ Processed ({len(text)} chars)"
            
        except Exception as e:
            return f"❌ Error: {e}"


def main():
    """Main extraction function"""
    print("🎯 PubMedBERT Text Feature Extraction")
    print("=" * 60)
    
    # Configuration
    csv_file = 'scc_survival_balanced_1000.csv'
    data_root = 'data/patients'
    
    # Load dataset
    df = pd.read_csv(csv_file)
    print(f"📊 Dataset: {len(df)} cases")
    
    # Initialize extractor
    extractor = TextFeatureExtractor()
    
    # Process all cases
    print(f"\n🔄 Processing {len(df)} cases...")
    
    results = {
        'processed': 0,
        'already_done': 0,
        'failed': 0,
        'no_text': 0
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text features"):
        case_id = row['ID']
        status = extractor.process_case(case_id, data_root)
        
        if "Already processed" in status:
            results['already_done'] += 1
        elif "✅ Processed" in status:
            results['processed'] += 1
        elif "No pathology" in status or "No text" in status or "Empty text" in status:
            results['no_text'] += 1
        else:
            results['failed'] += 1
        
        # Progress update every 100 cases
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{len(df)} cases")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Newly processed: {results['processed']}")
    print(f"⏭️  Already done: {results['already_done']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"📝 No text data: {results['no_text']}")
    print(f"📁 Total ready: {results['processed'] + results['already_done']}")
    
    success_rate = (results['processed'] + results['already_done']) / len(df) * 100
    print(f"🎯 Success rate: {success_rate:.1f}%")
    
    if success_rate > 90:
        print(f"\n🎉 Text feature extraction completed successfully!")
        print(f"💡 Features saved to: data/patients/*/processed_text_features/")
        print(f"🚀 Ready for 2.5-modal training!")
    else:
        print(f"\n⚠️  Some cases failed. Check individual case directories.")


def verify_extraction():
    """Verify extraction results"""
    print("\n🔍 Verifying text feature extraction...")
    
    df = pd.read_csv('scc_survival_balanced_1000.csv')
    sample_cases = df.head(5)['ID'].tolist()
    
    for case_id in sample_cases:
        feature_file = f'data/patients/{case_id}/processed_text_features/pubmedbert_features.npy'
        if os.path.exists(feature_file):
            features = np.load(feature_file)
            print(f"✅ {case_id}: {features.shape} features")
        else:
            print(f"❌ {case_id}: No features found")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_extraction()
    else:
        main()
        print(f"\n🔍 Run verification: python extract_text_features.py --verify")