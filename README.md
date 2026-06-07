# MCQ Finetune Embed

**Transfer Learning for Automated Distractor Effectiveness Assessment in Medical Multiple-Choice Questions: Fine-tuning Embedding Models to Predict Distractor Plausibility**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project investigates whether fine-tuned embedding models can predict distractor selection rates from textual features. The study establishes the technical feasibility of text-based distractor effectiveness prediction under a unified evaluation protocol across 10 embedding models.

### Key Results

Under a unified 5-fold cross-validation protocol (MSE loss, linear output, batch size 16):

- **Fine-tuning improved prediction** from frozen baselines to r = 0.627–0.644 across 8 of 10 models
- **Medical models**: SapBERT achieved r = 0.644 (+59.9%), BioLORD r = 0.635 (+62.0%)
- **General models**: BGE-base achieved r = 0.630 (+48.3%), BGE-large r = 0.626 (+34.0%)
- **Compact models**: MiniLM (22M) reached r = 0.627, MedEmbed-small (33M) reached r = 0.629
- **Lexical baselines**: TF-IDF with overlap features achieved r = 0.546; fine-tuned models exceeded this by +18.0%
- **Full fine-tuning** dramatically outperforms frozen encoder approaches
- **Cross-disciplinary generalizability** across 8 clinical specialties

## Problem Statement

Medical MCQ distractors must balance plausibility (attracting partially knowledgeable examinees) with discriminability (not confusing well-prepared examinees). Traditional quality assurance requires post-hoc response data analysis, which is resource-intensive and cannot guide item development proactively. This framework enables pre-deployment text-based screening of distractor effectiveness.

## Dataset

- **6,000 medical MCQs** from a national-level medical licensing assessment
- **23,999 distractor samples** with observed selection rates from prior test administrations
- **8 clinical disciplines**: Cardiology, Endocrinology, Haematology, Infectious Diseases, Nephrology, Neurology, Respiratory Medicine, and Rheumatology
- **Evaluation**: 5-fold cross-validation, stratified by discipline

### Data Format

```json
{
  "id": "Rheumatology_518",
  "question": "A 58-year-old man presents with...",
  "content_area": "Rheumatology",
  "options": [
    {
      "text": "Allopurinol initiation during the acute attack",
      "is_correct": false,
      "selection_rate": 0.28,
      "has_valid_text": true
    }
  ]
}
```

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for complete dataset documentation.

## Quick Start

### Installation

```bash
git clone https://github.com/zjiang4/MCQfinetuneEmbed.git
cd MCQfinetuneEmbed
pip install -r requirements.txt
```

### Pipeline

#### 1. Data Preprocessing

```bash
python scripts/01_preprocess_new_data.py
```

#### 2. Baseline Evaluation (General-Purpose Models)

```bash
python scripts/04_multi_model_baselines.py
```

#### 3. Extended Baseline (Medical Domain-Specific Models)

```bash
python scripts/15_extended_baseline.py
```

#### 4. Comprehensive Fine-tuning (BGE-large, MPNet)

```bash
python scripts/13_comprehensive_v2.py
```

#### 5. Fine-tune General Models (Contrastive Learning)

```bash
python scripts/05_finetune_all_models.py
```

#### 6. Fine-tune Medical Domain-Specific Models

```bash
python scripts/medical_finetune_fixed.py
```

#### 7. Statistical Analysis

```bash
python scripts/06_statistical_tests.py
python scripts/compute_revision_analyses.py
```

#### 8. Discipline-Specific Analysis

```bash
python scripts/17_discipline_analysis.py
```

#### 9. Ensemble Analysis

```bash
python scripts/18_ensemble_analysis.py
```

#### 10. Generate Figures

```bash
python scripts/generate_figures.py
python scripts/generate_medical_embedding_figures.py
```

## Project Structure

```
MCQfinetuneEmbed/
├── scripts/
│   ├── 01_preprocess_new_data.py          # Data preprocessing
│   ├── 04_multi_model_baselines.py        # General model baseline evaluation
│   ├── 05_finetune_all_models.py          # Fine-tune general models (contrastive)
│   ├── 06_statistical_tests.py            # Statistical significance tests
│   ├── 13_comprehensive_v2.py             # Comprehensive fine-tuning (MSE/MAE/Huber/Combined)
│   ├── 14_new_models_baseline.py          # Medical model baseline evaluation
│   ├── 15_extended_baseline.py            # Combined general + medical baselines
│   ├── 16_finetune_new_embeddings.py      # Fine-tune medical models (LoRA)
│   ├── 17_discipline_analysis.py          # Per-discipline performance analysis
│   ├── 18_ensemble_analysis.py            # Ensemble performance estimation
│   ├── medical_finetune_fixed.py          # Medical model fine-tuning (CosineSimilarityLoss)
│   ├── compute_revision_analyses.py       # Analytical revision statistics
│   ├── generate_figures.py                # Manuscript figure generation
│   └── generate_medical_embedding_figures.py  # Medical embedding figures
├── data/
│   └── processed/                         # Train/val/test splits (not included)
├── outputs/
│   └── results/                           # Experimental results
│       ├── comprehensive_v2/              # Table 2-7 data (48 configurations)
│       ├── baseline_all/                  # Table 1 baseline data
│       ├── extended_baseline/             # Medical model baselines
│       ├── medical_fixed/                 # Medical model fine-tuning results
│       ├── finetuned/                     # General model fine-tuning results
│       └── loco/                          # Leave-one-condition-out results
├── paper/
│   ├── npj_digital_medicine_v3_with_medical_embeddings.md  # Revised manuscript
│   ├── RESPONSE_TO_REVIEWERS.md           # Point-by-point response to reviewers
│   ├── SUPPLEMENTARY_MATERIALS.md         # Supplementary materials
│   └── REVISION_PLAN.md                   # Data consistency audit
├── figures/                               # Generated figures
├── DATA_DICTIONARY.md                     # Dataset documentation
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## Models Evaluated

### General-Purpose Models (5)

| Model | Parameters | Hidden Dim | Source |
|-------|-----------|------------|--------|
| BAAI/bge-large-en-v1.5 | 335M | 1024 | [HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| BAAI/bge-base-en-v1.5 | 109M | 768 | [HuggingFace](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| intfloat/e5-large-v2 | 335M | 1024 | [HuggingFace](https://huggingface.co/intfloat/e5-large-v2) |
| all-mpnet-base-v2 | 109M | 768 | [HuggingFace](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| all-MiniLM-L6-v2 | 22M | 384 | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

### Medical Domain-Specific Models (5)

| Model | Parameters | Hidden Dim | Pre-training |
|-------|-----------|------------|-------------|
| FremyCompany/BioLORD-2023 | 109M | 768 | PubMed + UMLS |
| ncbi/MedCPT-Query-Encoder | 109M | 768 | PubMed query-article pairs |
| ncbi/MedCPT-Article-Encoder | 109M | 768 | PubMed query-article pairs |
| abhinand/MedEmbed-small-v0.1 | 33M | 384 | Biomedical texts |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | 109M | 768 | PubMed full-text |

## Technical Approach

### Distractor-Targeted Fine-Tuning

The approach encodes question-distractor pairs and optimises for predicting the continuous selection rate metric:

- **Input format**: `'Question: {question_text} Option: {distractor_text}'`
- **Pooling**: CLS token embedding from the final transformer layer
- **Regression head**: Single linear layer mapping embeddings to scalar selection rates
- **Objective**: Minimise prediction error between predicted and observed selection rates

### Training Configurations (Unified Protocol)

All models were evaluated under a single unified configuration:
- **Loss function**: MSE
- **Output activation**: Linear (no sigmoid)
- **Batch size**: 16
- **Evaluation**: 5-fold CV on full dataset (N = 23,999), same splits
- **Early stopping**: Patience = 3 epochs on validation Pearson correlation

### Key Findings

1. **Full fine-tuning is essential**: dramatically outperforms frozen encoders
2. **Model convergence**: 6 models converge to r = 0.627–0.644 despite parameter range of 22M–335M
3. **Compact models are competitive**: MiniLM (22M) matches BGE-large (335M)
4. **Medical domain pre-training helps**: Medical models show larger relative improvements from fine-tuning
5. **Lexical features are strong but incomplete**: TF-IDF achieves r = 0.546; contextual models add +18.0%

## Citation

```bibtex
@article{jiang2026distractor,
  title={Transfer Learning for Automated Distractor Effectiveness Assessment in Medical Multiple-Choice Questions: Fine-tuning Embedding Models to Predict Distractor Plausibility},
  author={Jiang, Zhehan and Zheng, Tianpeng and Liu, Jiayi and Feng, Shicong},
  journal={Under Review},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Corresponding author: Zhehan Jiang (jiangzhehan@bjmu.edu.cn)
- GitHub Issues: [https://github.com/zjiang4/MCQfinetuneEmbed/issues](https://github.com/zjiang4/MCQfinetuneEmbed/issues)

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No. 72474004) and Peking University Health Science Center (Grant No. BMU2021YJ010).
