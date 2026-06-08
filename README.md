# MCQ Finetune Embed

**Transfer Learning for Text-Based Distractor Selection Rate Prediction in Medical Multiple-Choice Questions: Fine-tuning Embedding Models as a Plausibility Proxy**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This project investigates whether fine-tuned embedding models can predict distractor selection rates from textual features before test administration. Using 6,000 medical MCQs across eight clinical disciplines, we evaluate 10 embedding models (5 general-purpose, 5 medical domain-specific) under a unified 5-fold cross-validation protocol.

### Key Results

Under a unified 5-fold cross-validation protocol (MSE loss, linear output, batch size 16):

- **Fine-tuning improved prediction** from frozen baselines to r = 0.627–0.644 across 6 of 10 models
- **Medical models**: SapBERT achieved r = 0.644 (+59.9%), BioLORD r = 0.635 (+62.0%)
- **General models**: BGE-base achieved r = 0.630 (+48.3%), BGE-large r = 0.626 (+34.0%)
- **Compact models**: MiniLM (22M) reached r = 0.627, MedEmbed-small (33M) reached r = 0.629
- **Lexical baselines**: TF-IDF with overlap features achieved r = 0.546; fine-tuned models exceeded this by +18.0%
- **Cross-disciplinary generalizability** across 8 clinical specialties
- **Statistical significance**: Cohen's d > 1.0, all p < 0.001

## Problem Statement

Medical MCQ distractors must balance plausibility (attracting partially knowledgeable examinees) with discriminability (not confusing well-prepared examinees). Traditional quality assurance requires post-hoc response data analysis, which is resource-intensive and cannot guide item development proactively. This framework enables pre-deployment text-based screening of distractor plausibility using fine-tuned embedding models.

## Dataset

- **6,000 medical MCQs** from a national-level medical licensing assessment
- **23,999 distractor samples** with observed selection rates from prior test administrations
- **8 clinical disciplines**: Cardiology, Endocrinology, Haematology, Infectious Diseases, Nephrology, Neurology, Respiratory Medicine, and Rheumatology
- **Selection rate range**: 0.00–0.36 (M = 0.136, SD = 0.068)
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

### Demo Pipeline

An executable demo is provided that runs the full pipeline on 10 synthetic MCQ examples, requiring no private data or API keys:

```bash
python demo_pipeline.py
```

### Pipeline

#### 1. Data Preprocessing

```bash
python scripts/01_preprocess_new_data.py
```

#### 2. Unified Baseline Evaluation (All 10 Models)

```bash
python scripts/15_extended_baseline.py
```

#### 3. Unified Fine-tuning (MSE Loss, Linear Output)

```bash
python scripts/medical_finetune_fixed.py
python scripts/05_finetune_all_models.py
```

#### 4. Statistical Analysis

```bash
python scripts/06_statistical_tests.py
python scripts/compute_revision_analyses.py
```

#### 5. Qualitative Case Analysis

```bash
python scripts/qualitative_case_analysis.py
```

#### 6. Generate Figures

```bash
python scripts/generate_figures.py
```

## Project Structure

```
MCQfinetuneEmbed/
├── scripts/
│   ├── 01_preprocess_new_data.py              # Data preprocessing
│   ├── 05_finetune_all_models.py              # Fine-tune general models
│   ├── 06_statistical_tests.py                # Statistical significance tests
│   ├── 15_extended_baseline.py                # All-model baseline evaluation
│   ├── medical_finetune_fixed.py              # Medical model fine-tuning
│   ├── compute_revision_analyses.py           # Revision statistics
│   ├── qualitative_case_analysis.py           # Qualitative case study analysis
│   ├── generate_figures.py                    # Manuscript figure generation
│   └── generate_medical_embedding_figures.py  # Medical embedding figures
├── data/
│   └── processed/                             # Train/val/test splits (not included)
├── outputs/
│   └── results/                               # Experimental results
├── figures/                                   # Generated figures
├── demo_pipeline.py                           # Self-contained demo (R1)
├── DATA_DICTIONARY.md                         # Dataset documentation
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file
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

### Input Format

- **Format**: `'Question: {question_text} Option: {distractor_text}'`
- **Correct answer is intentionally excluded** from input, as pre-deployment screening should not require access to the correct answer
- **Pooling**: CLS token embedding from the final transformer layer

### Regression Architecture

```
Pre-trained Encoder → Linear(256) → ReLU → Dropout(0.1) → Output(1)
```

### Unified Training Protocol

All models were evaluated under a single unified configuration:
- **Loss function**: MSE
- **Output activation**: Linear (no sigmoid)
- **Batch size**: 16
- **Evaluation**: 5-fold CV on full dataset (N = 23,999), same stratified splits
- **Early stopping**: Patience = 3 epochs on validation Pearson correlation
- **Optimizer**: AdamW (lr = 2×10⁻⁵, weight decay = 0.01)
- **Gradient clipping**: Max norm 1.0
- **Random seeds**: Fixed (42) for PyTorch, NumPy, and Python

### Key Findings

1. **Full fine-tuning is essential**: dramatically outperforms frozen encoder approaches
2. **Model convergence**: 6 models converge to r = 0.627–0.644 despite parameter range of 22M–335M
3. **Compact models are competitive**: MiniLM (22M) matches BGE-large (335M)
4. **Medical domain pre-training helps**: Medical models show larger relative improvements from fine-tuning (+59.9% to +100.4% vs +34.0% to +104.1%)
5. **Lexical features are strong but incomplete**: TF-IDF achieves r = 0.546; contextual models add +18.0%
6. **Unstable models**: MedCPT-Query (SD = 0.273), MPNet (SD = 0.255), E5-large (SD = 0.247) showed high cross-fold variability under the unified protocol

## Citation

```bibtex
@article{jiang2026distractor,
  title={Transfer Learning for Text-Based Distractor Selection Rate Prediction in Medical Multiple-Choice Questions: Fine-tuning Embedding Models as a Plausibility Proxy},
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
