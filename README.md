# MCQ Finetune Embed

**Transfer Learning for Automated Distractor Effectiveness Assessment in Medical Multiple-Choice Questions: Fine-tuning Embedding Models to Predict Distractor Plausibility**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project implements a framework for automated pre-deployment screening of distractor effectiveness in medical MCQs using fine-tuned embedding models. By training pre-trained text embedding models on a distractor-targeted regression task, we predict the selection rate (proportion of examinees choosing each option) from textual features alone.

### Key Results

- **48.6% improvement** over baseline frozen embeddings (best fine-tuned: Pearson r = 0.653 vs. baseline r = 0.439, p < 0.001, Cohen's d = 1.19)
- **Cross-disciplinary generalizability** across 8 clinical specialties (r = 0.49-0.58 at baseline)
- **Medical domain-specific embeddings** achieve strong zero-shot performance (BioLORD-2023 and MedCPT-Query: r = 0.560, +11.1% vs. best general baseline)
- **Full fine-tuning** dramatically outperforms frozen encoder approaches (+93.8% for BGE-large)
- **Loss function robustness**: all loss functions achieve strong performance (r = 0.629-0.653)

## Problem Statement

Medical MCQ distractors must balance plausibility (attracting partially knowledgeable examinees) with discriminability (not confusing well-prepared examinees). Traditional quality assurance requires post-hoc response data analysis, which is resource-intensive and cannot guide item development proactively. This framework enables pre-deployment text-based screening of distractor effectiveness.

## Dataset

- **6,000 medical MCQs** from a national-level medical licensing assessment
- **16,800 distractor samples** with observed selection rates from prior test administrations
- **8 clinical disciplines**: Cardiology, Endocrinology, Haematology, Infectious Diseases, Nephrology, Neurology, Respiratory Medicine, and Rheumatology
- **Data split**: 4,200 train / 896 validation / 904 test (70/15/15 stratified by discipline)

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

### Training Configurations Evaluated

- **Loss Functions**: MSE, MAE, Huber Loss, Combined (MSE + Cosine Similarity)
- **Training Strategies**: Full fine-tuning (all parameters), Frozen encoder (regression head only)
- **Learning Rates**: 1x10^-5, 2x10^-5, 5x10^-5 (grid search)
- **Cross-validation**: 5-fold stratified cross-validation
- **Early stopping**: Patience = 3 epochs on validation Pearson correlation

### Key Findings

1. **Full fine-tuning is essential**: +93.8% improvement over frozen encoders for BGE-large
2. **Optimal learning rates scale inversely with model size**: 335M models need LR=1e-5, 22-109M models need LR=2e-5
3. **Loss function choice is secondary**: All achieve r = 0.629-0.653
4. **Medical embeddings offer strong zero-shot performance**: r = 0.560 without any fine-tuning
5. **Medical models are parameter-efficient**: MedCPT-Article (109M) achieves 97.6% of BGE-large (335M) performance after fine-tuning

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
