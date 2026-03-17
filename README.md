# MCQ Finetune Embed

**Transfer Learning for Pre-deployment Quality Assurance in Medical Education Assessment: Predicting Distractor Effectiveness via Fine-tuned Embeddings**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

This project implements an AI-driven framework for **automated pre-deployment quality assurance** in medical education assessment. By fine-tuning pre-trained embedding models on distractor-targeted tasks, we can predict the effectiveness of distractors (incorrect options) in multiple-choice questions (MCQs) before they are deployed in high-stakes examinations.

### Key Achievement

- **48.6% improvement** in distractor effectiveness prediction (Pearson r = 0.653 vs baseline r = 0.439, p < 0.001)
- **Strong cross-disciplinary generalizability** (r ≥ 0.60 across 8 clinical domains)
- **Parameter-efficient** approach using LoRA/DoRA for scalable deployment

## 🎯 Problem Statement

Medical education assessments play a critical role in ensuring patient safety. Poorly constructed distractors in MCQs may allow examinees with knowledge gaps to pass assessments, potentially compromising clinical decision-making. Traditional quality assurance methods are:

- **Reactive**: Analyze response data after exams are delivered
- **Resource-intensive**: Require substantial time and effort
- **Limited**: Provide minimal guidance for future item development

This framework enables **proactive** quality screening during item development, before exams are deployed.

## 📊 Dataset

- **6,000 medical MCQs** from 8 clinical disciplines
- **~16,800 distractor samples** with observed selection rates
- Disciplines: Cardiology, Endocrinology, Haematology, Infectious Diseases, Nephrology, Neurology, Respiratory, Rheumatology

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

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/zjiang4/MCQfinetuneEmbed.git
cd MCQfinetuneEmbed
pip install -r requirements.txt
```

### Usage

#### 1. Data Preprocessing

```bash
python 01_preprocess_new_data.py
```

#### 2. Baseline Experiments

```bash
python 02_baseline_experiments.py
```

#### 3. Fine-tuning Experiments

```bash
python 05_finetune_all_models.py
```

## 📁 Project Structure

```
MCQfinetuneEmbed/
├── 01_preprocess_new_data.py      # Data preprocessing and cleaning
├── 02_baseline_experiments.py      # Baseline model evaluation
├── 03_full_experiment.py           # Full experimental pipeline
├── 04_multi_model_baselines.py     # Multi-model baseline comparison
├── 05_finetune_all_models.py       # Fine-tuning all models
├── DATA_DICTIONARY.md              # Dataset documentation
├── README.md                       # This file
└── results/                        # Experimental results
    ├── baselines/                  # Baseline experiment outputs
    └── finetuned/                  # Fine-tuned model outputs
```

## 🔬 Models Evaluated

### General-Purpose Models
- BGE (BAAI General Embedding)
- GTE (General Text Embeddings)
- E5 (Embeddings from Bidirectional Encoder Representations)
- Instructor
- AnglE

### Medical Domain-Specific Models
- BioBERT
- ClinicalBERT
- PubMedBERT
- MedCPT
- BioLinkBERT

## 🛠️ Technical Approach

### Distractor-Targeted Fine-Tuning

Unlike standard fine-tuning that treats distractors as generic text, our approach:

1. **Models functional role**: Captures "plausible but incorrect" semantics
2. **Semantic relationship learning**: Balances proximity to correct answer with logical distinction
3. **Parameter-efficient**: Uses LoRA/DoRA for scalable deployment

### Training Configurations Evaluated

- **Loss Functions**: MSE, MAE, Huber Loss, Cosine Similarity
- **Training Strategies**: Full fine-tuning, LoRA, DoRA
- **Learning Rates**: Grid search optimization
- **Batch Sizes**: Memory-efficient configurations

## 📚 Research Questions Addressed

1. **Effectiveness**: Does distractor-targeted fine-tuning substantially improve quality prediction?
2. **Optimization**: What training configurations maximize prediction accuracy?
3. **Generalizability**: How do architectural factors influence cross-discipline performance?

## 🔗 Citation

If you use this code or dataset, please cite:

```bibtex
@article{mcq_finetune_embed_2026,
  title={Transfer Learning for Pre-deployment Quality Assurance in Medical Education Assessment: Predicting Distractor Effectiveness via Fine-tuned Embeddings},
  author={Jiang, Zhengjian and colleagues},
  journal={Under Review},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration opportunities:
- GitHub Issues: [https://github.com/zjiang4/MCQfinetuneEmbed/issues](https://github.com/zjiang4/MCQfinetuneEmbed/issues)

## 🙏 Acknowledgments

This research contributes to the broader goal of integrating AI into medical education assessment infrastructure, ultimately supporting the development of competent healthcare professionals and safeguarding patient safety.

---

**Note**: This framework is intended for research purposes and should be used as a decision-support tool by qualified medical educators, not as a replacement for expert judgment in assessment design.
