# MAGIC: Multimodal Analysis of Genomics, Imaging and Clinical Data

## 🎯 Overview
This repository supports the [MAGIC](https://magic.polygonhealthanalytics.com/) (Multimodal Analysis of Genomics, Imaging and Clinical Data) platform, enabling users to perform multimodal survival analysis on their own datasets.

In this tutorial, we demonstrate the workflow using an example of squamous cell carcinoma, integrating three key data modalities:
- **Whole Slide Images (WSI)**: Histopathology whole slide images in ".svs" format
- **RNA-Seq Data**: STAR-aligned gene-level read counts with annotation in ".tsv" format
- **Pathology Text**: Clinical narratives extracted from pathology reports in ".txt" format

![Workflow Overview](Figs/Fig1.png)
*Figure 1: Complete workflow flowchart showing the data processing and analysis pipeline*

## 📋 Prerequisites
Before running this analysis, you need to:
- Download multimodal data from MAGIC platform (clinical, RNA-Seq, pathology reports, and WSI data)
- Python 3.8+ with required packages
- GPU recommended for training (especially for WSI processing)
- Sufficient storage space (~1TB for processed features)

## 🚀 Quick Start
1. **Data Download**: Obtain SCC multimodal data from MAGIC (Refer to "Task 2: Add Complexity to Your Cohort" on the MAGIC user tutorial page)
2. **Setup Environment**: `pip install -r requirements.txt`
3. **Configuration**: Review and customize `configs/model_config.yaml` if needed
4. **Data Processing**: Run the preprocessing pipeline: `process_all_cases_smart.py` and `extract_text_features.py`.
5. **Model Training**: Execute the 3-modal survival analysis: `scc_3modal_training.py`
6. **Results Analysis**: Evaluate model performance: `generate_predictions_1347.py`

## 📊 Data Processing Pipeline

### Step 1: Data Download from MAGIC
**Note**: This step requires access to the MAGIC platform. Please ensure you have:
- Valid MAGIC platform credentials
- Access to TCGA SCC datasets
- Sufficient storage space

### Step 2: Complete Data Processing

1. **Pathology Text Processing**:
   - OCR Conversion: Amazon Textract extracts plain text from PDF pathology reports
   - Text Cleaning: Remove artifacts and standardize formatting
   - Feature Extraction: PubMedBERT encodes text into 768-dimensional feature vectors

2. **WSI Image Processing**:
   - Patch Selection: CLAM (Clustering-constrained Attention Multiple instance learning) selects informative patches from WSI
   - Feature Extraction: Pre-trained mSTAR WSI encoder extracts 2048-dimensional features from selected patches

3. **RNA-Seq Data Processing**:
   - Gene Expression: Gene-level counts obtained
   - Normalization: Standard preprocessing applied
   - Format: Compatible with the mSTAR framework

### Step 3: Data Organization
After processing, organize your data as follows:
```
data/
├── patients/
│   ├── TCGA-XXXX-XXXX/
│   │   ├── clinical/           # Clinical data (JSON)
│   │   ├── rnaseq/            # RNA-Seq data (TSV)
│   │   ├── pathology/         # Processed pathology text
│   │   ├── wsi/              # Whole slide images (SVS)
│   │   └── features/          # Extracted features
└── processed/                 # Final processed dataset
```

## 🏗️ Model Architecture

### Multimodal Fusion Framework
This analysis is built on the **mSTAR (Multimodal Survival Analysis)** framework, specifically using the `multimodal_survival` module from the mSTAR downstream tasks.

### Model Selection: Porpoise
We selected the **Porpoise model** for this analysis due to its characteristics:
- **RNA-Seq Tolerance**: Handling of RNA-Seq data without requiring signature files
- **Flexible Architecture**: Supports variable gene sets (we use 5000 genes vs MCAT's 753)
- **Bilinear Fusion**: Advanced fusion mechanism for multimodal integration

### Architecture Components
1. **WSI Branch**: Deep Sets + Attention mechanism
2. **RNA-Seq Branch**: SNN (Sparse Neural Network) blocks
3. **Text Branch**: PubMedBERT features
4. **Fusion**: Bilinear fusion of 256-dimensional features
5. **Output**: 4-class survival prediction

### Training Configuration
- **Loss Function**: NLLSurvLoss (Negative Log-Likelihood Survival Loss)
- **Optimizer**: Adam with learning rate 2e-4
- **Scheduler**: Cosine annealing
- **Evaluation**: 5-fold cross-validation with Concordance Index
- **Training**: 40 epochs with batch size 1

## 📈 Results

![Results Visualization](Figs/Fig2.png)
*Figure 2: Results visualization from the survival prediction analysis*

***Left***: Pie chart showing the contribution of each modality in the multimodal fusion model. ***Right***: Bar plot displaying C-index performance across 5-fold cross-validation. The 3-modal model achieves consistent performance across folds. The C-index 0.6961 ± 0.0170 is comparable to the mSTAR listed benchmark. 

## 🔧 Technical Implementation

### Core Dependencies
- **mSTAR Framework**: Multimodal survival analysis framework
- **CLAM**: Patch selection for WSI processing
- **PubMedBERT**: Pathology text feature extraction
- **Amazon Textract**: OCR for pathology reports

### Environment Setup
```bash
# Install required packages
pip install -r requirements.txt

# Or install individually:
pip install torch torchvision
pip install transformers  # For PubMedBERT
pip install openslide-python  # For WSI processing
pip install pandas numpy scikit-learn
pip install tqdm matplotlib seaborn
...
```

### YAML Configuration File
The `configs/model_config.yaml` file serves as a centralized configuration system for the entire analysis pipeline. 
- **Model Architecture**: Model type, fusion method, output classes
- **Data Modalities**: Feature dimensions for WSI, RNA-Seq, and text
- **Training Parameters**: Learning rate, epochs, optimizer settings
- **Evaluation**: Cross-validation folds, metrics
- **Hardware**: Device settings, number of workers

## 📚 Citations and Acknowledgments

This use case is based on:

```bibtex
@article{mstar2023,
  title={mSTAR: Multimodal Survival Analysis Framework},
  author={...},
  journal={...},
  year={2023}
}

@article{clam2020,
  title={CLAM: Clustering-constrained Attention Multiple instance learning},
  author={Lu, M.Y. and Williamson, D.F.K. and Chen, T.Y. and Chen, R.J. and Barbieri, M. and Mahmood, F.},
  journal={Nature Machine Intelligence},
  volume={2},
  pages={369--378},
  year={2020}
}

@article{pubmedbert2019,
  title={PubMedBERT: A Domain-Specific Language Model for Biomedical Text},
  author={Gu, Y. and Tinn, R. and Cheng, H. and Lucas, M. and Usuyama, N. and Liu, X. and Naumann, T. and Gao, J. and Poon, H.},
  journal={arXiv preprint arXiv:2007.15779},
  year={2020}
}
```

### MAGIC Platform
This analysis demonstrates the capabilities of the [MAGIC platform](https://magic.polygonhealthanalytics.com/) for multimodal cancer data analysis. MAGIC provides unified access to diverse cancer datasets and enables integrative analysis workflows.

## 🤝 Contributing
This repository serves as a demonstration of MAGIC platform capabilities. For questions about the MAGIC platform or data access, please refer to the official MAGIC documentation.

## 🔗 Related Links
- [mSTAR Framework](https://github.com/Innse/mSTAR)
- [CLAM Repository](https://github.com/mahmoodlab/CLAM)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)
- [TCGA Data Portal](https://portal.gdc.cancer.gov/)

## 📝 Notes
- This repository contains only the analysis code and does not include raw data
- Data must be downloaded separately from the MAGIC platform
- Results may vary depending on the specific dataset and preprocessing steps 
