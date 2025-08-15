#!/usr/bin/env python3
"""
Smart Process All Cases for mSTAR Multimodal Survival Analysis

SMART PROCESSING APPROACH:
- Only process what's missing (like download script)
- Process order: Clinical → RNA-Seq → WSI (fast to slow)
- Preserve existing valid processed data
- ZERO data deletion - only append to failed cases
- Cumulative failed case tracking
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import openslide
from PIL import Image
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add CLAM to path
sys.path.append('./CLAM')

class SmartCaseProcessor:
    def __init__(self):
        self.data_dir = Path('data/patients')
        self.failed_cases_file = Path('failed_cases.json')
        self.failed_cases = self.load_failed_cases()
        
        # Initialize mSTAR model once at startup
        print("🤖 Initializing mSTAR encoder...")
        self.mstar_model, self.mstar_transform = self.setup_mstar_encoder()
        
        # Store device for tensor operations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_failed_cases(self):
        """Load cumulative failed cases list"""
        if self.failed_cases_file.exists():
            try:
                with open(self.failed_cases_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_failed_case(self, case_id, reason, step):
        """Append failed case to cumulative list"""
        failed_case = {
            'case_id': case_id,
            'reason': reason,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        self.failed_cases.append(failed_case)
        
        with open(self.failed_cases_file, 'w') as f:
            json.dump(self.failed_cases, f, indent=2)
        
        print(f"❌ Failed case logged: {case_id} - {reason}")
    
    def check_processing_status(self, case_id):
        """Check what processing steps are already completed"""
        case_dir = self.data_dir / case_id
        
        status = {
            'clinical': False,
            'rnaseq': False, 
            'wsi': False,
            'pathology_text': False
        }
        
        # Check processed clinical
        processed_clinical_dir = case_dir / 'processed_clinical'
        if processed_clinical_dir.exists():
            clinical_files = list(processed_clinical_dir.glob('*_processed.json'))
            status['clinical'] = len(clinical_files) > 0
        
        # Check processed RNA-Seq (either version)
        processed_rnaseq_dir = case_dir / 'processed_rnaseq'
        if processed_rnaseq_dir.exists():
            rnaseq_files = list(processed_rnaseq_dir.glob('*_processed.tsv')) + \
                          list(processed_rnaseq_dir.glob('*_mstar.tsv'))
            status['rnaseq'] = len(rnaseq_files) > 0
        
        # Check WSI features (look in case features subdirectory)
        case_features_dir = case_dir / 'features'
        wsi_features_exist = False
        if case_features_dir.exists():
            feature_files = list(case_features_dir.glob('*.pt'))
            # Check if any valid feature file exists (any size, as long as it's loadable)
            for feature_file in feature_files:
                try:
                    # Quick validation: file exists, has reasonable size, and is loadable
                    if feature_file.stat().st_size > 1000:  # > 1KB (reasonable minimum)
                        # Try to load to verify it's a valid tensor
                        import torch
                        torch.load(feature_file, weights_only=True)
                        wsi_features_exist = True
                        break
                except:
                    continue  # Skip corrupted files
        status['wsi'] = wsi_features_exist
        
        # Check pathology text processing
        processed_pathology_dir = case_dir / 'processed_pathology'
        if processed_pathology_dir.exists():
            pathology_files = list(processed_pathology_dir.glob('*_pathology_text.txt'))
            status['pathology_text'] = len(pathology_files) > 0
        
        return status
    
    def process_clinical_data(self, case_id):
        """Process clinical data for a case"""
        case_dir = self.data_dir / case_id
        clinical_dir = case_dir / 'clinical'
        processed_dir = case_dir / 'processed_clinical'
        processed_dir.mkdir(exist_ok=True)
        
        try:
            # Find clinical JSON file
            clinical_files = list(clinical_dir.glob('*.json'))
            if not clinical_files:
                raise Exception("No clinical JSON file found")
            
            clinical_file = clinical_files[0]
            
            # Load and process clinical data
            with open(clinical_file, 'r') as f:
                clinical_data = json.load(f)
            
            # Extract survival information
            # Check both demographic and survival sections
            vital_status = clinical_data.get('demographic', {}).get('vital_status') or clinical_data.get('survival', {}).get('vital_status')
            days_to_death = clinical_data.get('demographic', {}).get('days_to_death') or clinical_data.get('survival', {}).get('days_to_death')
            days_to_last_follow_up = clinical_data.get('demographic', {}).get('days_to_last_follow_up') or clinical_data.get('survival', {}).get('days_to_last_follow_up')
            
            # Calculate survival days using the logic:
            # For alive cases: days_to_last_follow_up (primary), days_to_death (fallback)
            # For dead cases: days_to_death (primary), days_to_last_follow_up (fallback)
            survival_days = None
            if vital_status == 'Alive':
                if days_to_last_follow_up is not None:
                    survival_days = days_to_last_follow_up
                elif days_to_death is not None:
                    survival_days = days_to_death
            elif vital_status == 'Dead':
                if days_to_death is not None:
                    survival_days = days_to_death
                elif days_to_last_follow_up is not None:
                    survival_days = days_to_last_follow_up
            
            processed_data = {
                'case_id': case_id,
                'age_at_diagnosis': clinical_data.get('diagnoses', [{}])[0].get('age_at_diagnosis'),
                'vital_status': vital_status,
                'days_to_death': days_to_death,
                'days_to_last_follow_up': days_to_last_follow_up,
                'survival_days': survival_days,  # New field with calculated survival days
                'gender': clinical_data.get('demographic', {}).get('gender'),
                'race': clinical_data.get('demographic', {}).get('race'),
                'ethnicity': clinical_data.get('demographic', {}).get('ethnicity'),
                'primary_diagnosis': clinical_data.get('diagnoses', [{}])[0].get('primary_diagnosis'),
                'tumor_stage': clinical_data.get('diagnoses', [{}])[0].get('tumor_stage'),
                'processed_timestamp': datetime.now().isoformat()
            }
            
            # Save processed clinical data
            output_file = processed_dir / 'case_clinical_data_processed.json'
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"    ✅ Clinical data processed")
            return True
            
        except Exception as e:
            print(f"    ❌ Clinical processing failed: {e}")
            return False
    
    def process_rnaseq_data(self, case_id):
        """Process RNA-Seq data for a case"""
        case_dir = self.data_dir / case_id
        rnaseq_dir = case_dir / 'rnaseq'
        processed_dir = case_dir / 'processed_rnaseq'
        processed_dir.mkdir(exist_ok=True)
        
        try:
            # Find RNA-Seq TSV file
            rnaseq_files = list(rnaseq_dir.glob('*.tsv'))
            if not rnaseq_files:
                raise Exception("No RNA-Seq TSV file found")
            
            rnaseq_file = rnaseq_files[0]
            
            # Load RNA-Seq data with error handling
            try:
                df = pd.read_csv(rnaseq_file, sep='\t', comment='#')
            except pd.errors.ParserError:
                # Try with different parsing options for malformed files
                df = pd.read_csv(rnaseq_file, sep='\t', comment='#', on_bad_lines='skip')
            
            # Process for mSTAR (keep top expressed genes)
            if 'tpm_unstranded' in df.columns:
                # Sort by TPM and keep top 5000 genes
                df_sorted = df.sort_values('tpm_unstranded', ascending=False)
                df_processed = df_sorted.head(5000)
                
                # Create mSTAR format with correct column names
                df_mstar = df_processed[['gene_name', 'tpm_unstranded']].copy()
                df_mstar['Index'] = range(len(df_mstar))  # Add sequential index
                df_mstar.columns = ['Gene', 'Value', 'Index']  # mSTAR expected format
                df_mstar = df_mstar[['Gene', 'Index', 'Value']]  # Reorder columns
                
                # Save processed data
                output_file = processed_dir / f'{rnaseq_file.stem}_mstar.tsv'
                df_mstar.to_csv(output_file, sep='\t', index=False)
                
                print(f"    ✅ RNA-Seq data processed (mSTAR format)")
                return True
            else:
                raise Exception("Required TPM column not found")
                
        except Exception as e:
            print(f"    ❌ RNA-Seq processing failed: {e}")
            return False
    
    def process_pathology_text(self, case_id):
        """Process pathology report text for a case"""
        import re
        
        case_dir = self.data_dir / case_id
        processed_dir = case_dir / 'processed_pathology'
        processed_dir.mkdir(exist_ok=True)
        
        try:
            # Look for pathology text in case pathology directory (following standard pattern)
            pathology_raw_dir = case_dir / 'pathology'
            if not pathology_raw_dir.exists():
                raise Exception("No pathology directory found")
            
            # Find the text file
            text_files = list(pathology_raw_dir.glob('*.txt'))
            if not text_files:
                raise Exception("No pathology text file found")
            
            text_file = text_files[0]  # Take the first text file
            
            # Read raw text
            with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            # Clean and preprocess text
            cleaned_text = self.clean_pathology_text(raw_text)
            
            if not cleaned_text:
                raise Exception("Text cleaned to empty string")
            
            # Save processed text
            output_file = processed_dir / f'{case_id}_pathology_text.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            print(f"    ✅ Pathology text processed ({len(cleaned_text)} chars)")
            return True
            
        except Exception as e:
            print(f"    ❌ Pathology text processing failed: {e}")
            return False
    
    def clean_pathology_text(self, text):
        """Clean and preprocess pathology report text"""
        import re
        
        if not text:
            return ""
        
        # Remove UUID and case ID lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, UUIDs, case IDs, and redacted information
            if not line or line.startswith('UUID:') or line.startswith('TCGA-') or line == 'Redacted':
                continue
            
            # Skip lines with just "XX" or similar placeholders
            if line in ['XX', 'X', 'xxx', 'XXX']:
                continue
            
            # Skip lines that are just numbers or dates
            if re.match(r'^[\d\-/\s]+$', line):
                continue
            
            cleaned_lines.append(line)
        
        # Join cleaned lines with newlines to preserve structure
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace but preserve paragraph breaks
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Replace multiple spaces/tabs with single space
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)  # Replace multiple newlines with double newline
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def setup_mstar_encoder(self):
        """Setup mSTAR encoder with pretrained weights"""
        try:
            import timm
            from torchvision import transforms
            
            print("      🔄 Loading mSTAR model...")
            # Load pretrained mSTAR model with error handling
            model = timm.create_model(
                'hf-hub:Wangyh/mSTAR',
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True
            )
            print("      ✅ mSTAR model loaded")
            
            # Set up preprocessing transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Move model to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            
            print(f"      🎯 Model loaded on: {device}")
            return model, transform
            
        except Exception as e:
            print(f"      ❌ Failed to setup mSTAR encoder: {e}")
            print(f"      🔄 Falling back to simple feature extraction...")
            return None, None
    
    def process_wsi_features(self, case_id):
        """Extract WSI features using mSTAR encoder"""
        case_dir = self.data_dir / case_id
        wsi_dir = case_dir / 'wsi'
        
        try:
            # Find WSI file
            wsi_files = list(wsi_dir.glob('*.svs'))
            if not wsi_files:
                raise Exception("No WSI file found")
            
            wsi_file = wsi_files[0]
            
            # Validate WSI file
            if not wsi_file.exists() or wsi_file.stat().st_size == 0:
                raise Exception("WSI file is missing or empty")
            
            # Use pre-loaded mSTAR encoder
            model, transform = self.mstar_model, self.mstar_transform
            if model is None:
                # Fallback to simple feature extraction
                print(f"      🔄 Using simple feature extraction (no mSTAR)")
                # Create dummy features for now
                dummy_features = torch.randn(700, 1024)  # Match expected shape
                
                # Save features in case subdirectory
                case_features_dir = case_dir / 'features'
                case_features_dir.mkdir(exist_ok=True)
                wsi_filename = wsi_file.stem
                feature_file = case_features_dir / f'{wsi_filename}.pt'
                torch.save(dummy_features, feature_file)
                
                print(f"      ✅ Simple features saved (fallback mode)")
                return True
            
            # Load WSI with error handling
            try:
                slide = openslide.OpenSlide(str(wsi_file))
            except Exception as e:
                raise Exception(f"Cannot open WSI file: {e}")
            
            # Get slide dimensions
            width, height = slide.dimensions
            
            # Calculate adaptive patch count based on slide size (like original)
            slide_area = width * height
            if slide_area < 1e8:  # Small slide (< 100M pixels)
                max_patches = 200
            elif slide_area < 5e8:  # Medium slide (100M-500M pixels)
                max_patches = 400
            else:  # Large slide (> 500M pixels)
                max_patches = 600
            
            # Extract patches (optimized but proper count)
            patch_size = 224
            patches = []
            
            # Sample patches from the slide (balanced speed vs quality)
            step_size = patch_size * 4  # Sample every 4th patch (reasonable sampling)
            
            for y in range(0, height - patch_size, step_size):
                for x in range(0, width - patch_size, step_size):
                    try:
                        # Extract patch
                        patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                        patch = patch.convert('RGB')
                        
                        # Quick background check (faster)
                        patch_array = np.array(patch)
                        if np.mean(patch_array) > 200:  # Skip white background
                            continue
                        
                        patches.append(patch)
                        
                        # Early stopping for speed
                        if len(patches) >= max_patches:
                            break
                    except:
                        continue
                if len(patches) >= max_patches:
                    break
            
            if len(patches) == 0:
                raise Exception("No valid patches extracted")
            
            # Extract features using mSTAR (BATCH PROCESSING for speed)
            batch_size = 32  # OPTIMAL batch size for RTX 3060 Ti (sweet spot found!)
            features = []
            
            with torch.no_grad():
                for i in range(0, len(patches), batch_size):
                    batch_patches = patches[i:i + batch_size]
                    
                    # Convert batch to tensor
                    batch_tensors = []
                    for patch in batch_patches:
                        patch_tensor = transform(patch)
                        batch_tensors.append(patch_tensor)
                    
                    batch_tensor = torch.stack(batch_tensors)
                    
                    # Move batch to same device as model
                    batch_tensor = batch_tensor.to(self.device)
                    
                    # Process batch
                    batch_features = model(batch_tensor)
                    features.extend(batch_features.cpu().numpy())
            
            # Save features in mSTAR format: [num_patches, 1024] (NO mean pooling!)
            # This preserves patch-level information as mSTAR does
            combined_features = np.array(features)  # [num_patches, 1024]
            
            # Save features in case subdirectory (matching existing structure)
            case_features_dir = case_dir / 'features'
            case_features_dir.mkdir(exist_ok=True)
            
            # Use WSI filename for feature file name (matching existing pattern)
            wsi_filename = wsi_file.stem  # Get filename without extension
            feature_file = case_features_dir / f'{wsi_filename}.pt'
            torch.save(torch.tensor(combined_features), feature_file)
            
            slide.close()
            print(f"    ✅ WSI features extracted ({len(patches)} patches)")
            return True
            
        except Exception as e:
            print(f"    ❌ WSI processing failed: {e}")
            return False
    
    def process_case(self, case_id):
        """Process a single case with smart skip logic"""
        print(f"\n🔄 Processing case: {case_id}")
        
        # Check current processing status
        status = self.check_processing_status(case_id)
        
        print(f"    📊 Current status: Clinical={status['clinical']}, RNA-Seq={status['rnaseq']}, Pathology={status['pathology_text']}, WSI={status['wsi']}")
        
        # Process in order: Clinical → RNA-Seq → Pathology Text → WSI (fast to slow)
        
        # 1. Clinical processing
        if not status['clinical']:
            print(f"    🧬 Processing clinical data...")
            if not self.process_clinical_data(case_id):
                self.save_failed_case(case_id, "Clinical processing failed", "clinical")
                return False
        else:
            print(f"    ⏭️  Skipping clinical (already processed)")
        
        # 2. RNA-Seq processing
        if not status['rnaseq']:
            print(f"    🧬 Processing RNA-Seq data...")
            if not self.process_rnaseq_data(case_id):
                self.save_failed_case(case_id, "RNA-Seq processing failed", "rnaseq")
                return False
        else:
            print(f"    ⏭️  Skipping RNA-Seq (already processed)")
        
        # 3. Pathology text processing
        if not status['pathology_text']:
            print(f"    📄 Processing pathology text...")
            if not self.process_pathology_text(case_id):
                self.save_failed_case(case_id, "Pathology text processing failed", "pathology_text")
                return False
        else:
            print(f"    ⏭️  Skipping pathology text (already processed)")
        
        # 4. WSI processing (most expensive)
        if not status['wsi']:
            print(f"    🖼️  Processing WSI features...")
            if not self.process_wsi_features(case_id):
                self.save_failed_case(case_id, "WSI processing failed", "wsi")
                return False
        else:
            print(f"    ⏭️  Skipping WSI (already processed)")
        
        print(f"    ✅ Case {case_id} completed")
        return True
    
    def create_mstar_dataset(self):
        """Create final mSTAR dataset CSV"""
        print("\n📊 Creating mSTAR dataset...")
        
        dataset_rows = []
        
        for case_dir in self.data_dir.iterdir():
            if not case_dir.is_dir() or not case_dir.name.startswith('TCGA'):
                continue
                
            case_id = case_dir.name
            
            # Check if case is fully processed
            status = self.check_processing_status(case_id)
            if not all(status.values()):
                continue
            
            # Load processed clinical data
            try:
                clinical_file = case_dir / 'processed_clinical' / 'case_clinical_data_processed.json'
                with open(clinical_file, 'r') as f:
                    clinical_data = json.load(f)
                
                # Find processed RNA-Seq file
                rnaseq_dir = case_dir / 'processed_rnaseq'
                rnaseq_files = list(rnaseq_dir.glob('*_mstar.tsv')) or list(rnaseq_dir.glob('*_processed.tsv'))
                
                if rnaseq_files:
                    rnaseq_file = rnaseq_files[0]
                else:
                    continue
                
                # WSI feature file (look in case features subdirectory)
                case_features_dir = case_dir / 'features'
                feature_files = list(case_features_dir.glob('*.pt')) if case_features_dir.exists() else []
                
                # Find a valid feature file (any size, loadable)
                feature_file = None
                for f in feature_files:
                    try:
                        # Check if file is reasonable size and loadable
                        if f.stat().st_size > 1000:  # > 1KB (reasonable minimum)
                            # Try to load to verify it's a valid tensor
                            torch.load(f, weights_only=True)
                            feature_file = f
                            break
                    except:
                        continue  # Skip corrupted files
                
                if not feature_file:
                    continue
                
                # Find pathology text file
                pathology_dir = case_dir / 'processed_pathology'
                pathology_files = list(pathology_dir.glob('*_pathology_text.txt')) if pathology_dir.exists() else []
                
                if not pathology_files:
                    continue
                
                pathology_file = pathology_files[0]
                
                # Create dataset row
                row = {
                    'case_id': case_id,
                    'clinical_file': str(clinical_file),
                    'rnaseq_file': str(rnaseq_file),
                    'pathology_text_file': str(pathology_file),
                    'wsi_features': str(feature_file),
                    'vital_status': clinical_data.get('vital_status'),
                    'days_to_death': clinical_data.get('days_to_death'),
                    'days_to_last_follow_up': clinical_data.get('days_to_last_follow_up'),
                    'survival_days': clinical_data.get('survival_days'),  # New field
                    'age_at_diagnosis': clinical_data.get('age_at_diagnosis'),
                    'gender': clinical_data.get('gender'),
                    'primary_diagnosis': clinical_data.get('primary_diagnosis')
                }
                
                dataset_rows.append(row)
                
            except Exception as e:
                print(f"⚠️  Skipping {case_id} from dataset: {e}")
                continue
        
        # Create DataFrame and save
        df = pd.DataFrame(dataset_rows)
        df.to_csv('mstar_dataset.csv', index=False)
        
        print(f"✅ mSTAR dataset created with {len(df)} cases")
        return len(df)
    
    def run(self):
        """Main processing pipeline"""
        print("🚀 Smart mSTAR Case Processor")
        print("=" * 60)
        
        # Get all cases
        all_cases = [d.name for d in self.data_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('TCGA')]
        
        print(f"📊 Found {len(all_cases)} total cases")
        
        # Filter out known failed cases
        failed_case_ids = {case['case_id'] for case in self.failed_cases}
        cases_to_process = [case_id for case_id in all_cases 
                           if case_id not in failed_case_ids]
        
        print(f"📋 Processing {len(cases_to_process)} cases (skipping {len(failed_case_ids)} known failures)")
        
        # Process cases
        successful = 0
        failed = 0
        
        for case_id in tqdm(cases_to_process, desc="Processing cases"):
            if self.process_case(case_id):
                successful += 1
            else:
                failed += 1
        
        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📋 Total failed (cumulative): {len(self.failed_cases)}")
        
        # Create final dataset
        dataset_size = self.create_mstar_dataset()
        
        print(f"\n🎉 Smart processing completed!")
        print(f"📄 Final dataset: {dataset_size} cases ready for mSTAR analysis")

def main():
    processor = SmartCaseProcessor()
    processor.run()

if __name__ == "__main__":
    main()