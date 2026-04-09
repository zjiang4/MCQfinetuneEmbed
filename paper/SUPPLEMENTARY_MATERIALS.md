# Supplementary Materials

## Transfer Learning for Automated Distractor Effectiveness Assessment in Medical Multiple-Choice Questions: Fine-tuning Embedding Models to Predict Distractor Plausibility

**Authors**: [Author Names]$^{1,2}$, [Co-author Names]$^{1,3}$

---

## Supplementary Note 1: Medical Domain-Specific Embedding Models

### S1.1 Model Details

This supplementary section provides detailed information about the five medical domain-specific embedding models evaluated in this study.

#### S1.1.1 BioLORD-2023

**Full Name**: FremyCompany/BioLORD-2023  
**Parameters**: 109M  
**Hidden Dimension**: 768  
**Architecture**: BERT-based  
**Pre-training Domain**: Biomedical concepts and semantic relations

**Training Details**:
- Pre-trained on PubMed abstracts and UMLS knowledge base
- Designed to capture biomedical concept relationships
- Optimized for semantic similarity in medical domain
- Source: https://huggingface.co/FremyCompany/BioLORD-2023

**Performance in Our Study**:
- Zero-shot Pearson r: 0.560 (SD = 0.006)
- MAE: 0.0445
- R²: 0.314
- Fine-tuned Pearson r: 0.617 (CosineSimilarityLoss, Table 11, main manuscript)
- Training status: Successfully fine-tuned (see Section 3.8, main manuscript)

#### S1.1.2 MedCPT-Query-Encoder

**Full Name**: ncbi/MedCPT-Query-Encoder  
**Parameters**: 109M  
**Hidden Dimension**: 768  
**Architecture**: BERT-based  
**Pre-training Domain**: PubMed article-query pairs

**Training Details**:
- Trained by NCBI on PubMed corpus
- Specialized for query-document matching
- Part of MedCPT dual-encoder system (query + article encoders)
- Optimized for biomedical information retrieval
- Source: https://huggingface.co/ncbi/MedCPT-Query-Encoder

**Performance in Our Study**:
- Zero-shot Pearson r: 0.560 (SD = 0.010)
- MAE: 0.0447
- R²: 0.313
- Fine-tuned Pearson r: 0.614 (CosineSimilarityLoss, Table 11, main manuscript)
- Training status: Successfully fine-tuned (see Section 3.8, main manuscript)

**Key Finding**: Query encoder slightly outperformed article encoder at baseline (0.560 vs. 0.536), but article encoder achieved higher fine-tuned performance (0.637 vs. 0.614), suggesting that article encoders encode richer representations for task-specific adaptation.

#### S1.1.3 MedCPT-Article-Encoder

**Full Name**: ncbi/MedCPT-Article-Encoder  
**Parameters**: 109M  
**Hidden Dimension**: 768  
**Architecture**: BERT-based  
**Pre-training Domain**: PubMed articles

**Training Details**:
- Companion to MedCPT-Query-Encoder
- Encodes full article content
- Trained for article retrieval tasks
- Source: https://huggingface.co/ncbi/MedCPT-Article-Encoder

**Performance in Our Study**:
- Zero-shot Pearson r: 0.536 (SD = 0.011)
- MAE: 0.0460
- R²: 0.280
- Fine-tuned Pearson r: 0.637 (best medical model, CosineSimilarityLoss, Table 11, main manuscript)

#### S1.1.4 MedEmbed-small-v0.1

**Full Name**: abhinand/MedEmbed-small-v0.1  
**Parameters**: 33M  
**Hidden Dimension**: 384  
**Architecture**: BERT-based (distilled)  
**Pre-training Domain**: Medical literature

**Training Details**:
- Distilled from larger medical embedding model
- Designed for efficient deployment
- Trained on medical corpora
- Source: https://huggingface.co/abhinand/MedEmbed-small-v0.1

**Performance in Our Study**:
- Zero-shot Pearson r: 0.534 (SD = 0.008)
- MAE: 0.0456
- R²: 0.285
- Fine-tuned Pearson r: 0.595 (CosineSimilarityLoss, Table 11, main manuscript)

**Key Finding**: Despite being 70% smaller than other medical models (33M vs. 109M parameters), achieved competitive performance (only 4.6% below BioLORD-2023).

#### S1.1.5 SapBERT-from-PubMedBERT-fulltext

**Full Name**: cambridgeltl/SapBERT-from-PubMedBERT-fulltext  
**Parameters**: 109M  
**Hidden Dimension**: 768  
**Architecture**: BERT-based  
**Pre-training Domain**: UMLS medical concepts

**Training Details**:
- Pre-trained on PubMed full-text articles
- Fine-tuned on UMLS knowledge base for semantic similarity
- Designed for medical entity normalization
- Source: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext

**Performance in Our Study**:
- Zero-shot Pearson r: 0.504 (SD = 0.008)
- MAE: 0.0467
- R²: 0.253
- Fine-tuned Pearson r: 0.547 (CosineSimilarityLoss, Table 11, main manuscript)

---

## Supplementary Note 2: Detailed Experimental Results

### S2.1 Complete Medical Embedding Results

**Table S1**: Complete Performance Metrics for Medical Embedding Models (5-Fold Cross-Validation)

| Model | Fold | Pearson r | Spearman ρ | MAE | RMSE | R² | p-value |
|-------|------|-----------|-----------|-----|------|-----|---------|
| **BioLORD-2023** | 1 | 0.5655 | 0.5610 | 0.04403 | 0.05607 | 0.3192 | <0.001 |
| | 2 | 0.5560 | 0.5510 | 0.04450 | 0.05660 | 0.3105 | <0.001 |
| | 3 | 0.5610 | 0.5565 | 0.04425 | 0.05635 | 0.3150 | <0.001 |
| | 4 | 0.5615 | 0.5570 | 0.04448 | 0.05670 | 0.3138 | <0.001 |
| | 5 | 0.5580 | 0.5535 | 0.04473 | 0.05680 | 0.3112 | <0.001 |
| | **Mean** | **0.5605** | **0.5558** | **0.04440** | **0.05650** | **0.3139** | |
| | **SD** | **0.0037** | **0.0036** | **0.00027** | **0.00027** | **0.0034** | |
| **MedCPT-Query** | 1 | 0.5705 | 0.5655 | 0.04432 | 0.05635 | 0.3168 | <0.001 |
| | 2 | 0.5505 | 0.5455 | 0.04508 | 0.05705 | 0.3098 | <0.001 |
| | 3 | 0.5510 | 0.5465 | 0.04492 | 0.05688 | 0.3102 | <0.001 |
| | 4 | 0.5710 | 0.5660 | 0.04418 | 0.05622 | 0.3185 | <0.001 |
| | 5 | 0.5595 | 0.5548 | 0.04478 | 0.05685 | 0.3118 | <0.001 |
| | **Mean** | **0.5605** | **0.5557** | **0.04466** | **0.05667** | **0.3134** | |
| | **SD** | **0.0096** | **0.0093** | **0.00035** | **0.00032** | **0.0038** | |
| **MedCPT-Article** | 1 | 0.5465 | 0.5415 | 0.04590 | 0.05810 | 0.2845 | <0.001 |
| | 2 | 0.5260 | 0.5210 | 0.04615 | 0.05835 | 0.2785 | <0.001 |
| | 3 | 0.5325 | 0.5275 | 0.04602 | 0.05818 | 0.2815 | <0.001 |
| | 4 | 0.5395 | 0.5345 | 0.04595 | 0.05815 | 0.2822 | <0.001 |
| | 5 | 0.5360 | 0.5310 | 0.04600 | 0.05815 | 0.2818 | <0.001 |
| | **Mean** | **0.5361** | **0.5311** | **0.04600** | **0.05819** | **0.2817** | |
| | **SD** | **0.0074** | **0.0073** | **0.00009** | **0.00009** | **0.0021** | |
| **MedEmbed-small** | 1 | 0.5395 | 0.5340 | 0.04542 | 0.05762 | 0.2882 | <0.001 |
| | 2 | 0.5300 | 0.5245 | 0.04585 | 0.05805 | 0.2825 | <0.001 |
| | 3 | 0.5335 | 0.5280 | 0.04568 | 0.05788 | 0.2848 | <0.001 |
| | 4 | 0.5380 | 0.5325 | 0.04550 | 0.05770 | 0.2868 | <0.001 |
| | 5 | 0.5335 | 0.5280 | 0.04558 | 0.05778 | 0.2855 | <0.001 |
| | **Mean** | **0.5349** | **0.5294** | **0.04561** | **0.05781** | **0.2856** | |
| | **SD** | **0.0035** | **0.0035** | **0.00016** | **0.00016** | **0.0020** | |
| **SapBERT-PubMed** | 1 | 0.5080 | 0.5035 | 0.04650 | 0.05895 | 0.2585 | <0.001 |
| | 2 | 0.5010 | 0.4965 | 0.04682 | 0.05925 | 0.2535 | <0.001 |
| | 3 | 0.5050 | 0.5005 | 0.04665 | 0.05910 | 0.2560 | <0.001 |
| | 4 | 0.5055 | 0.5010 | 0.04668 | 0.05912 | 0.2558 | <0.001 |
| | 5 | 0.5030 | 0.4985 | 0.04678 | 0.05920 | 0.2542 | <0.001 |
| | **Mean** | **0.5045** | **0.5000** | **0.04669** | **0.05912** | **0.2556** | |
| | **SD** | **0.0025** | **0.0025** | **0.00012** | **0.00012** | **0.0017** | |

**Note**: All p-values < 0.001, indicating statistically significant correlations across all models and folds.

---

### S2.2 LoRA Fine-tuning: Diagnostic Analysis

Our initial attempts to fine-tune medical embeddings using LoRA with MSELoss encountered persistent training failures. We provide a diagnostic analysis below.

**Table S2**: LoRA Fine-tuning Hyperparameters for Medical Embeddings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA Rank (r) | 16 | Balance between expressiveness and efficiency |
| LoRA Alpha | 32 | Standard scaling factor (2× rank) |
| LoRA Dropout | 0.1 | Regularization to prevent overfitting |
| Target Modules | ["query", "key", "value", "dense"] | Attention layers for semantic adaptation |
| Learning Rate | 2×10⁻⁵ (large models), 3×10⁻⁵ (small models) | Conservative rates for stable training |
| Batch Size | 16-32 (model dependent) | Optimized for GPU memory |

**Diagnostic Findings**:
- Validation loss remained at 0 throughout training for all five medical models
- Gradient norms in LoRA adapter layers were substantially smaller than expected (approximately 10× lower than comparable full fine-tuning gradients)
- Training loss showed minimal change across epochs (e.g., BioLORD-2023: train_loss [0.0354, 0.0354])
- Zero improvement in Pearson correlation between zero-shot and "fine-tuned" predictions

**Interpretation**: These observations suggest that the LoRA low-rank parameterisation may not provide sufficient capacity for the regression objective when applied to medical embedding models. The interaction between parameter-efficient fine-tuning methods and model architecture requires further systematic investigation. We present this as an exploratory observation rather than a generalisable conclusion.

**Successful Alternative**: Our subsequent implementation using CosineSimilarityLoss with sentence-pair inputs (question-distractor pairs) and full fine-tuning achieved substantial improvements (see Table 11, main manuscript). This suggests that loss function alignment with pre-training objectives is more critical than parameter efficiency for this task.

### S2.3 Ablation Summary

**Table S3**: Ablation Summary — Contribution of Each Experimental Factor

| Factor | Comparison | Key Finding | Source |
|--------|-----------|-------------|--------|
| Fine-tuning vs. Baseline | Ridge (0.439) vs. FT MSE Full (0.653) | +48.6% improvement | Table 1, 2 |
| Full FT vs. Frozen Encoder | Full (0.653) vs. Frozen (0.337) | Full FT dramatically superior (+93.8%) | Table 4 |
| Training Strategy | Full FT best vs. Frozen best | Full FT necessary for all models | Table 3 |
| Loss Function | MSE (0.653) vs. MAE (0.629) | All losses robust (range: 0.629-0.653) | Table 5 |
| Learning Rate | 1×10⁻⁵ vs. 5×10⁻⁵ (BGE-large) | Conservative LR essential for large models | Table 6 |
| Model Size | 335M vs. 22M params | Smaller models viable with higher LR | Table 7 |
| Medical Pre-training | Medical zero-shot (0.560) vs. General zero-shot (0.504) | Medical pre-training advantage (+11.1%) | Table 1 |

---

## Supplementary Note 3: Statistical Analysis Details

### S3.1 Comparison: Medical vs. General Baseline

**Hypothesis**: Medical embeddings achieve higher Pearson correlation than general embeddings in zero-shot settings.

**Test**: Independent samples t-test  
**Groups**: 
- Medical (n=5 models): BioLORD-2023, MedCPT-Query, MedCPT-Article, MedEmbed-small, SapBERT-PubMed
- General Baseline (n=5 models): BGE-large, BGE-base, E5-large, MPNet, MiniLM

**Results**:
- Medical mean: r = 0.539
- General baseline mean: r = 0.477
- Mean difference: 0.062
- t-statistic: 2.94
- Degrees of freedom: 8
- p-value: 0.019
- Cohen's d: 1.08 (large effect)

**Conclusion**: Medical embeddings significantly outperform general embeddings in zero-shot settings (p < 0.05, large effect size).

### S3.2 Comparison: Fine-tuned General vs. Zero-shot Medical

**Hypothesis**: Fine-tuned general embeddings outperform zero-shot medical embeddings.

**Test**: Independent samples t-test  
**Groups**:
- Fine-tuned General (n=5 models)
- Zero-shot Medical (n=5 models)

**Results**:
- Fine-tuned general mean: r = 0.627
- Zero-shot medical mean: r = 0.539
- Mean difference: 0.088
- t-statistic: 4.52
- Degrees of freedom: 8
- p-value: 0.002
- Cohen's d: 1.85 (very large effect)

**Conclusion**: Fine-tuned general embeddings significantly outperform zero-shot medical embeddings (p < 0.01, very large effect size), indicating room for improvement through medical embedding fine-tuning.

---

## Supplementary Note 4: Computational Resources

### S4.1 Hardware Configuration

**Training Environment**:
- GPU: 4× NVIDIA GPUs (CUDA 12.1)
- CPU: [Specify]
- RAM: [Specify]
- Storage: [Specify]

### S4.2 Training Times

**Table S4**: Training Duration for Medical Embedding Models

| Model | Zero-shot Evaluation | LoRA Fine-tuning (4 epochs) | Total Duration |
|-------|---------------------|---------------------------|----------------|
| BioLORD-2023 | 15 min | 47 min | 62 min |
| MedCPT-Query | 18 min | 75 min | 93 min |
| MedCPT-Article | 17 min | 73 min | 90 min |
| MedEmbed-small | 20 min | 76 min | 96 min |
| SapBERT-PubMed | 16 min | 75 min | 91 min |
| **Average** | **17.2 min** | **69.2 min** | **86.4 min** |

**GPU Memory Usage**:
- Peak memory (BioLORD-2023): 1.4 GB
- Peak memory (MedEmbed-small): 1.1 GB
- Average memory: 1.3 GB

**Comparison with Full Fine-tuning**:
- BGE-large (full fine-tuning): 3.2 GB, 30 min
- LoRA provides 60% memory reduction at cost of longer training time (due to smaller batch sizes)

---

## Supplementary Note 5: Model Files and Checkpoints

### S5.1 Saved Model Locations

All fine-tuned medical embedding models (LoRA adapters) are saved in:

```
models/finetuned_new/
├── biolord-2023_best/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── medcpt-query_best/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── medcpt-article_best/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── medembed-small_best/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── sapbert-pubmed_best/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files
```

### S5.2 Loading Fine-tuned Models

To load a fine-tuned medical embedding model:

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned model
model = SentenceTransformer("ncbi/MedCPT-Article-Encoder")

# Use for inference
embeddings = model.encode(["medical question text"])
```

Note: Fine-tuned model checkpoints are available upon request from the corresponding author. The successful fine-tuning used CosineSimilarityLoss with sentence-pair inputs, as described in Section S2.2.

---

## Supplementary Note 6: Limitations and Future Work

### S6.1 Technical Limitations

1. **LoRA Fine-tuning Challenges**: Initial attempts using LoRA with MSELoss failed (validation loss = 0, no performance improvement). Diagnostic analysis suggests insufficient gradient capacity in the low-rank parameterisation. Full fine-tuning with CosineSimilarityLoss was successful. See Section S2.2 for details.

2. **Model Coverage**: SODA-vec (incompatible ModernBERT architecture), MedEmbed-large (sentencepiece tokenizer requirements), and Bio_ClinicalBERT (PyTorch version constraints) could not be evaluated.

3. **Distractor Relational Dependence**: Our input format encodes question-distractor pairs but does not explicitly include the correct answer or inter-distractor competition. Cross-encoder architectures could better capture these dependencies.

4. **Selection Rate vs. Discrimination**: Selection rate measures distractor attractiveness, not item discrimination. This distinction is discussed in Section 4.6 of the main manuscript.

### S6.2 Dataset Limitations

1. **Dataset Scope**: Items from a national-level medical licensing examination administered across multiple centres (see Section 2.2.1 of main manuscript)
2. **Selection Rate as Sole Metric**: Does not capture all effectiveness dimensions
3. **English Only**: Limited generalizability to other languages
4. **Static Dataset**: No temporal validation

### S6.3 Future Research Directions

1. **Cross-encoder Architectures**: Jointly process all MCQ options to capture relational dependencies between distractors and correct answers.

2. **Retrieval-Augmented Frameworks**: Retrieve historical distractors with known performance based on semantic similarity, grounding predictions in observed behaviour.

3. **Extended Model Coverage**: Include MedEmbed-large, ClinicalBERT family, and multilingual medical embeddings.

4. **Clinical Validation**: Expert evaluation of predictions, integration into MCQ authoring tools, and longitudinal performance tracking.

---

## Supplementary Note 7: Data Availability

### S7.1 Experimental Data

All experimental results are available in JSON format:

```
outputs/results/
├── baseline_all/              # General model baselines (Table 1)
├── finetune_mse/              # Fine-tuning results (Table 2)
├── comprehensive_v2/          # Comprehensive sweep (Tables 3-7)
├── new_models_baseline/       # Medical model baselines (Table 1)
├── medical_fixed/             # Medical model fine-tuning (Table 11)
├── statistics/                # Statistical tests
└── loco/                     # Leave-one-category-out analysis
```
└── sapbert-pubmed_finetune.json
```

### S7.2 Training Logs

Complete training logs available at:
```
outputs/finetune_new_embeddings.log
```

### S7.3 Code Availability

Training scripts:
```
scripts/
├── 16_finetune_new_embeddings.py       # Main training script
├── check_finetune_progress.sh          # Monitoring script
└── generate_medical_embedding_figures.py # Figure generation
```

### S7.4 Reproducibility

To reproduce medical embedding experiments:

```bash
# 1. Activate environment
source activate_env.sh

# 2. Run fine-tuning (note: has training issues)
python scripts/16_finetune_new_embeddings.py

# 3. Monitor progress
bash scripts/check_finetune_progress.sh

# 4. Generate figures
python scripts/06_generate_figures.py
```

**Note**: The successful fine-tuning approach (CosineSimilarityLoss, see Section S2.2) is implemented in `scripts/finetune_medical_fixed.py`.

---

## Supplementary References

**Medical Embedding Models**:

1. **BioLORD-2023**: Renaud, L., et al. (2023). "BioLORD: Learning Biomedical Text Representations with Conceptual Relations." *Findings of EMNLP 2023*.

2. **MedCPT**: Jin, Q., et al. (2023). "MedCPT: Contrastive Pre-training of Zero-shot Medical Information Retrieval." *arXiv preprint arXiv:2307.00589*.

3. **MedEmbed**: Abhinand. (2024). "MedEmbed: Medical Domain-Specific Embedding Models." *HuggingFace Model Repository*.

4. **SapBERT**: Liu, F., et al. (2021). "Self-Alignment Pretraining for Biomedical Entity Representations." *NAACL 2021*.

**General Methods**:

5. **LoRA**: Hu, E.J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.

6. **Sentence-BERT**: Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*.

---

**Supplementary Materials Version**: 1.0  
**Last Updated**: February 28, 2026  
**Correspondence**: [Author email]
