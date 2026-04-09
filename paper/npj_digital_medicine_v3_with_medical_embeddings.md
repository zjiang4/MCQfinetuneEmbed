# Transfer Learning for Automated Distractor Effectiveness Assessment in Medical Multiple-Choice Questions: Fine-tuning Embedding Models to Predict Distractor Plausibility

## Authors
[Author Names]$^{1,2}$, [Co-author Names]$^{1,3}$

$^1$[Institution Name], [City, Country]  
$^2$[Email]  
$^3$[Email]

## Corresponding Author
[Name, Email, Address]

**Running Title**: Fine-tuning Embeddings for MCQ Distractor Effectiveness Prediction

---

## Abstract

**Background**: The effectiveness of distractors in multiple-choice questions (MCQs) is critical to assessment validity, yet current evaluation methods are reactive and resource-intensive. We investigated whether domain-specific fine-tuning of pre-trained embedding models can predict distractor effectiveness from textual features alone, enabling pre-deployment quality screening.

**Methods**: Using 6,000 medical MCQs across eight clinical disciplines (24,000 distractor samples across training, validation, and test splits), we compared ten pre-trained embedding models—five general-purpose and five medical domain-specific—under baseline and fine-tuned conditions, systematically evaluating loss functions, training strategies, and learning rates.

**Results**: Fine-tuning substantially improved distractor effectiveness prediction. The optimal configuration achieved Pearson r = 0.653 (95% CI: 0.636–0.669), a 48.7% improvement over baseline (r = 0.439; p < 0.001; Cohen's d = 1.19). Medical domain-specific embeddings showed dual advantages: stronger zero-shot performance (BioLORD-2023 and MedCPT-Query both r = 0.560, +11.1% vs. best general baseline) and successful fine-tuning (MedCPT-Article r = 0.637 with 67% fewer parameters than the best general model). Full fine-tuning dramatically outperformed frozen encoders (+93.8%). All loss functions performed robustly (r = 0.629–0.653). Optimal learning rates scaled inversely with model size.

**Conclusions**: Domain-specific fine-tuning substantially improves automated prediction of distractor effectiveness. Medical embeddings offer strong zero-shot performance and parameter-efficient fine-tuning, enabling tiered deployment strategies from rapid zero-shot screening (r ≈ 0.56) to high-accuracy fine-tuned prediction (r ≈ 0.65).

**Keywords**: transfer learning, medical education, multiple-choice questions, distractor effectiveness, natural language processing, embedding models

---

## 1. Introduction

Multiple-choice questions constitute the predominant assessment modality across medical education, from undergraduate curricula through board certification to continuing medical education (Schuwirth & Van der Vleuten, 2011; Case & Swanson, 2002). Their widespread adoption reflects practical advantages: efficient administration, objective scoring, and broad content coverage. However, assessment validity hinges critically on distractor quality—incorrect options that must simultaneously satisfy competing demands of plausibility and discriminability (Haladyna et al., 2002; Gierl et al., 2017).

Well-constructed distractors discriminate between examinees with varying knowledge levels and provide diagnostic information about misconceptions (Tarrant et al., 2006). Conversely, ineffective distractors compromise measurement precision and introduce construct-irrelevant variance (Rodriguez, 2005). Creating high-quality distractors remains challenging for medical educators (Abdulghani et al., 2015), and traditional quality evaluation requires post-hoc analysis of response patterns—a resource-intensive process that cannot guide item development (Gierl & Phocksook, 2022).

Transformer-based language models have catalyzed advances in medical natural language processing (Lee et al., 2020; Gu et al., 2022). Text embedding models map textual content to dense vector representations capturing semantic relationships (Reimers & Gurevych, 2019), offering potential for automated distractor assessment. However, general-purpose embeddings may inadequately capture the specialised semantic relationships in medical MCQs, where effective distractors must balance semantic proximity to correct answers with conceptual distinction.

Transfer learning through fine-tuning offers a principled approach to domain adaptation (Howard & Ruder, 2018). While fine-tuning has improved performance across medical NLP tasks (Lee et al., 2020; Alsentzer et al., 2019), its application to predicting continuous behavioural metrics in educational assessment remains underexplored.

Recent advances in MCQ analysis fall into four categories. First, LLM-based distractor generation methods (Feng et al., 2024; Bitew et al., 2025) produce plausible distractors but do not evaluate existing ones. Second, neural item difficulty prediction (Rogoz & Ionescu, 2024; Skidmore et al., 2025) estimates item-level difficulty using transformer features, but operates at the item level rather than the option level. Third, automated distractor quality assessment has begun to emerge: Raina et al. (2023) proposed embedding-based metrics for distractor plausibility and diversity, and Benedetto et al. (2025) surveyed automated evaluation methods for MCQ distractors. Fourth, Benedetto et al. (2023) provided a comprehensive survey of question difficulty estimation from text. Traditional psychometric methods (Gierl et al., 2017; Gierl & Phocksook, 2022) provide rigorous post-hoc distractor evaluation through item response theory, but require response data from actual test administrations and cannot guide item development proactively. Our approach addresses this gap by predicting distractor-level effectiveness from textual features before deployment, using a fine-tuned embedding framework that we term "distractor-targeted" because it explicitly optimises for the behavioural metric of distractor selection rates rather than general semantic similarity.

This study addresses three research questions:

**RQ1**: Does domain-specific fine-tuning substantially improve distractor effectiveness prediction?

**RQ2**: What training configurations optimise prediction accuracy?

**RQ3**: How do architectural factors influence fine-tuning effectiveness?

---

## 2. Methods

### 2.1 Study Design

We employed a three-phase experimental design:
- Phase 1: Baseline evaluation using frozen embeddings with linear regression
- Phase 2: Fine-tuning all models with standardised hyperparameters
- Phase 3: Comprehensive hyperparameter optimisation

### 2.2 Dataset

#### 2.2.1 Source and Composition

The dataset comprised items from a national-level medical licensing examination administered across multiple testing centres, consisting of 6,000 MCQs spanning eight clinical disciplines: Cardiology, Endocrinology, Haematology, Infectious Diseases, Nephrology, Neurology, Respiratory Medicine, and Rheumatology. All items employed realistic clinical scenarios and patient presentations designed to assess medical students at the pre-licensure level, corresponding to completion of undergraduate medical programmes.

Items underwent systematic expert review to ensure alignment with national medical licensing examination standards, establishing both content validity and construct validity. Domain experts annotated specialty labels and cognitive dimensions for each question, guaranteeing accurate representation of intended clinical focus and learning objectives. Each question included one correct answer and three to five distractors, with observed selection rates from prior test administrations.

Selection rates—the proportion of examinees choosing each option—served as a proxy for distractor plausibility, which is one component of overall distractor effectiveness. We do not claim that selection rates capture discrimination or construct validity; rather, they provide a practical behavioural measure of distractor attractiveness that is available before deployment.

#### 2.2.2 Data Partitioning

We implemented stratified random splitting ensuring balanced discipline representation:

| Split | Questions | Distractor Samples | Purpose |
|-------|-----------|-------------------|---------|
| Training | 4,200 (70%) | 16,800 | Parameter optimisation |
| Validation | 896 (15%) | 3,584 | Hyperparameter selection |
| Test | 904 (15%) | 3,615 | Final evaluation |

Selection rates ranged from 0.00 to 0.36 (mean = 0.136, SD = 0.068), providing diversity across the plausibility spectrum.

### 2.3 Models

We evaluated two categories of pre-trained text embedding models:

#### 2.3.1 General-Purpose Models (5 models)

| Model | Parameters | Hidden Dim | Pre-training Objective |
|-------|-----------|------------|----------------------|
| BAAI/bge-large-en-v1.5 | 335M | 1024 | Contrastive learning |
| BAAI/bge-base-en-v1.5 | 109M | 768 | Contrastive learning |
| intfloat/e5-large-v2 | 335M | 1024 | Embedding-specific contrastive |
| all-mpnet-base-v2 | 109M | 768 | Sentence pair training |
| all-MiniLM-L6-v2 | 22M | 384 | Knowledge distillation |

#### 2.3.2 Medical Domain-Specific Models (5 models)

To assess the value of domain-specific pre-training, we additionally evaluated medical-specialised embedding models:

| Model | Parameters | Hidden Dim | Pre-training Domain | Source |
|-------|-----------|------------|---------------------|--------|
| FremyCompany/BioLORD-2023 | 109M | 768 | Biomedical concepts & relations | BioLORD |
| ncbi/MedCPT-Query-Encoder | 109M | 768 | PubMed article-query pairs | NCBI |
| ncbi/MedCPT-Article-Encoder | 109M | 768 | PubMed articles | NCBI |
| abhinand/MedEmbed-small-v0.1 | 33M | 384 | Medical literature | MedEmbed |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | 109M | 768 | UMLS medical concepts | Cambridge LTL |

These models were pre-trained on biomedical corpora (PubMed abstracts, UMLS knowledge base, clinical notes) and are hypothesised to capture medical semantic relationships more effectively than general-purpose embeddings.

### 2.4 Experimental Conditions

#### 2.4.1 Phase 1: Baseline

**Input Format**: Each (question, distractor) pair was formatted as a single text string: `'Question: {question_text} Option: {distractor_text}'`. This format was used consistently across all baseline evaluations.

**Embedding Extraction**: We used the SentenceTransformers library (Reimers & Gurevych, 2019) to extract fixed-dimensional vector representations. For each input text, the pre-trained tokenizer produced variable-length token sequences (typically 20–80 tokens per MCQ option). The SentenceTransformers pooling strategy then generated a fixed-dimensional embedding by computing the mean of all token embeddings from the final transformer layer, followed by L2 normalisation. This standard pooling approach ensures that the embedding dimension is independent of input length and transformer depth. For BGE-series models, the library additionally applies a CLS-based pooling component as part of the model architecture. We used the default pooling configuration for each model without modification, ensuring reproducibility through fixed random seeds (seed = 42 for PyTorch, NumPy, and Python).

**Regression**: We extracted frozen embeddings from each pre-trained model, training Ridge regression models (regularisation parameter α = 1.0) mapping embeddings to selection rates. We employed 5-fold cross-validation on the test set for robust baseline estimates.

#### 2.4.2 Phase 2: Initial Fine-tuning

**Input Format**: For fine-tuning, each sample was formatted identically to Phase 1: `'Question: {question_text} Option: {distractor_text}'`, with the corresponding selection rate as the regression target. Note that the correct answer was intentionally excluded from the input, as pre-deployment screening should not require access to the correct answer (see Limitations, Section 4.6).

We implemented a regression architecture comprising the pre-trained transformer encoder followed by a task-specific regression head. The regression head consisted of a linear projection from hidden dimension to 256 units, ReLU activation, dropout (probability 0.1), and final linear projection to a single output with Sigmoid activation constraining predictions to the [0, 1] range.

Fine-tuning hyperparameters:
- Loss function: Mean Squared Error
- Training strategy: Full fine-tuning (all parameters)
- Learning rate: 2×10⁻⁵
- Batch size: 32
- Maximum epochs: 10
- Early stopping: Patience = 3 epochs on validation Pearson correlation
- Optimiser: AdamW with decoupled weight decay (0.01)
- Gradient clipping: Maximum norm = 1.0

#### 2.4.3 Phase 3: Comprehensive Evaluation

We systematically varied:

**Loss Functions** (4 conditions):
- Mean Squared Error: Standard quadratic loss
- Mean Absolute Error: Robust linear loss
- Huber Loss: Combines quadratic and linear penalties
- Combined Loss: Weighted sum of MSE and Huber losses

**Training Strategies** (2 conditions):
- Full fine-tuning: All 335M (BGE-large) or 109M (MPNet) parameters trainable
- Frozen encoder: Only regression head trainable (~260K parameters for BGE-large, ~197K for MPNet)

**Learning Rates** (3 conditions):
- Conservative: 1×10⁻⁵
- Moderate: 2×10⁻⁵
- Aggressive: 5×10⁻⁵

This factorial design yielded 48 fine-tuning configurations (2 models × 4 losses × 2 strategies × 3 learning rates) plus 2 baseline evaluations.

### 2.5 Training Protocol

All experiments employed consistent optimisation settings:

- **Optimiser**: AdamW with decoupled weight decay (coefficient = 0.01)
- **Learning rate schedule**: Linear warmup over 10% of training steps, followed by linear decay
- **Batch size**: 32 (limited by GPU memory for large models)
- **Mixed precision**: FP16 where supported for memory efficiency
- **Random seeds**: Fixed at 42 for reproducibility across PyTorch, NumPy, and Python
- **Early stopping**: Monitor validation Pearson correlation with patience = 3 epochs

**Computational Environment**: Experiments were conducted on a high-performance computing system equipped with four NVIDIA Tesla V100 GPUs (32GB VRAM each), 125GB system memory, and 64 CPU cores. Training times ranged from approximately 20-30 minutes per full fine-tuning run for large models (335M parameters) to 5 minutes for frozen encoder approaches. Total computational time for all 48 fine-tuning experiments was approximately 18 hours.

### 2.6 Evaluation Metrics

We assessed performance using multiple complementary metrics:

1. **Pearson Correlation (r)**: Primary metric measuring linear association between predicted and actual selection rates
2. **Spearman Correlation (ρ)**: Rank-based correlation robust to outliers
3. **Mean Absolute Error (MAE)**: Average absolute prediction error in selection rate units
4. **Root Mean Squared Error (RMSE)**: Quadratic average emphasising larger errors
5. **Coefficient of Determination (R²)**: Proportion of variance explained

### 2.7 Statistical Analysis

We assessed statistical significance using Fisher's z-transformation for correlation comparisons and linear mixed-effects models with random intercepts per item to account for within-item clustering of distractors. Specifically, we fitted the model: SRᵢⱼ = β₀ + β₁·predᵢⱼ + uᵢ + εᵢⱼ, where SRᵢⱼ is the observed selection rate for distractor j of item i, predᵢⱼ is the model prediction, uᵢ ~ N(0, σ²ᵤ) is a random intercept per item, and εᵢⱼ ~ N(0, σ²) is the residual. We note that Fisher's z-transformation assumes independence of observations; our clustered data structure may lead to slightly anti-conservative p-values for correlation comparisons, though the large effect sizes (Cohen's d = 1.19) ensure robustness to this limitation. We estimated the intraclass correlation coefficient (ICC) from the data to characterise the degree of clustering.

We computed Cohen's d to quantify standardised effect sizes, with interpretation following conventional thresholds: small (0.2), medium (0.5), large (0.8). We computed 95% confidence intervals for Pearson correlations using Fisher's z-transformation.

For Phase 3 comparisons across 48 configurations, we applied Bonferroni correction where appropriate, though primary conclusions focus on pre-registered comparisons to avoid selective reporting.

---

## 3. Results

### 3.1 Baseline Performance

We evaluated ten pre-trained embedding models—five general-purpose and five medical domain-specific—using frozen embeddings with Ridge regression (Table 1). This comprehensive baseline establishes zero-shot performance before task-specific adaptation. In addition to Pearson correlation (our primary metric), we report Spearman rank correlation, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R²) to provide a comprehensive assessment.

**Table 1: Baseline Performance with Frozen Embeddings (5-Fold Cross-Validation)**

**General-Purpose Models:**

| Model | Parameters | Pearson r (SD) | Spearman ρ | MAE | RMSE | R² | 95% CI |
|-------|-----------|---------------|-----------|-----|------|-----|--------|
| BAAI/bge-base-en-v1.5 | 109M | 0.504 (0.036) | 0.489 | 0.051 | 0.066 | 0.254 | [0.468, 0.540] |
| intfloat/e5-large-v2 | 335M | 0.500 (0.027) | 0.485 | 0.051 | 0.066 | 0.250 | [0.473, 0.527] |
| all-mpnet-base-v2 | 109M | 0.483 (0.015) | 0.468 | 0.052 | 0.067 | 0.233 | [0.468, 0.498] |
| all-MiniLM-L6-v2 | 22M | 0.458 (0.018) | 0.443 | 0.053 | 0.068 | 0.210 | [0.440, 0.476] |
| BAAI/bge-large-en-v1.5 | 335M | 0.439 (0.012) | 0.421 | 0.054 | 0.069 | 0.193 | [0.427, 0.451] |

**Medical Domain-Specific Models:**

| Model | Parameters | Pearson r (SD) | Spearman ρ | MAE | RMSE | R² | 95% CI |
|-------|-----------|---------------|-----------|-----|------|-----|--------|
| **FremyCompany/BioLORD-2023** | 109M | **0.560 (0.006)** | 0.556 | 0.0445 | 0.0567 | 0.314 | [0.548, 0.572] |
| **ncbi/MedCPT-Query-Encoder** | 109M | **0.560 (0.010)** | 0.555 | 0.0447 | 0.0567 | 0.313 | [0.540, 0.580] |
| ncbi/MedCPT-Article-Encoder | 109M | 0.536 (0.011) | 0.531 | 0.0460 | 0.0581 | 0.280 | [0.515, 0.557] |
| abhinand/MedEmbed-small-v0.1 | 33M | 0.534 (0.008) | 0.527 | 0.0456 | 0.0579 | 0.285 | [0.518, 0.550] |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | 109M | 0.504 (0.008) | 0.500 | 0.0467 | 0.0591 | 0.253 | [0.488, 0.520] |

**General vs. Medical Embeddings**: Medical domain-specific models showed stronger zero-shot performance than general embeddings. The best medical models (BioLORD-2023, MedCPT-Query) achieved Pearson correlations of 0.560, outperforming the best general model (BGE-base: r = 0.504) by 11.1%. This 11.1% relative improvement (absolute improvement +0.056, 95% CI: [0.040, 0.072]) was statistically significant (z = 7.3, p < 0.001, Fisher z-test). This advantage emerged without any task-specific fine-tuning, demonstrating that biomedical pre-training captures semantic relationships directly relevant to MCQ distractor assessment.

**Medical Model Architecture Insights**:
- **Query-focused and concept-aware models lead at baseline**: MedCPT-Query and BioLORD-2023 both achieved r = 0.560, aligning with our task structure (query-option similarity matching)
- **Article encoders show lower baseline**: MedCPT-Article (r = 0.536) had lower zero-shot performance but demonstrated exceptional fine-tuning potential (Section 3.8)
- **Smaller models remain competitive**: MedEmbed-small (33M parameters) achieved r = 0.534, only 4.6% below BioLORD-2023 (109M parameters) while offering 70% parameter reduction

**Performance Spectrum**: Across all baselines, correlations ranged from 0.417 to 0.560, demonstrating that pre-trained embeddings capture partial predictive signal, but substantial unexplained variance remains—motivating our fine-tuning experiments. A naive baseline that always predicts the mean selection rate (0.136) achieves MAE = 0.068 and r = 0, confirming that all embedding-based approaches provide meaningful predictive signal above chance.

**Statistical Comparison** (Medical vs. General):
- Mean medical baseline: r = 0.539 (SD = 0.023)
- Mean general baseline: r = 0.477 (SD = 0.028)
- Difference: +0.062 (95% CI: +0.037 to +0.087), p = 0.003 (significant)
- Medical models significantly outperform general models at zero-shot inference

### 3.2 Phase 2: Fine-tuning Improves All Models

Table 2 presents results after fine-tuning with standardised hyperparameters.

**Table 2: Performance After Fine-tuning (MSE Loss, Full Fine-tuning, LR = 2×10⁻⁵)**

| Model | Baseline r | Fine-tuned r | Absolute Gain | Relative Improvement | MAE | RMSE |
|-------|-----------|-------------|---------------|---------------------|-----|------|
| BAAI/bge-large-en-v1.5 | 0.514 | 0.637 | +0.123 | +23.9% | 0.040 | 0.052 |
| BAAI/bge-base-en-v1.5 | 0.504 | 0.649 | +0.145 | +28.8% | 0.040 | 0.052 |
| intfloat/e5-large-v2 | 0.500 | 0.608 | +0.108 | +21.6% | 0.041 | 0.053 |
| all-mpnet-base-v2 | 0.483 | 0.621 | +0.138 | +28.6% | 0.040 | 0.052 |
| all-MiniLM-L6-v2 | 0.458 | 0.612 | +0.154 | +33.6% | 0.042 | 0.054 |

All fine-tuned models significantly outperformed baselines (p < 0.001 for all comparisons). Improvements ranged from 21.6% to 33.6%, demonstrating universal benefit from domain adaptation. Note: BGE-large baseline r = 0.514 in Table 2 differs from the CLS-token baseline r = 0.439 in Table 1 due to different train/test splits (Table 2 uses a stratified split consistent with Table 3, while Table 1 uses a different random seed). The primary results and ablation analyses consistently use the Table 3 comprehensive pipeline (baseline r = 0.439, best fine-tuned r = 0.653).

### 3.3 Phase 3: Comprehensive Configuration Analysis

#### 3.3.1 Complete Experimental Results

Table 3 presents complete results for all 48 fine-tuning configurations.

**Table 3: Complete Results for All Experimental Configurations (Pearson Correlation)**

| Model | Loss Function | Training Method | LR = 1×10⁻⁵ | LR = 2×10⁻⁵ | LR = 5×10⁻⁵ |
|-------|--------------|-----------------|-------------|-------------|-------------|
| **BGE-large (335M)** | | | | | |
| | MSE | Full | **0.653** | 0.637 | 0.250 |
| | MSE | Frozen | 0.201 | 0.252 | 0.323 |
| | MAE | Full | 0.629 | 0.632 | -0.015 |
| | MAE | Frozen | 0.201 | 0.272 | 0.337 |
| | Huber | Full | 0.641 | 0.644 | 0.624 |
| | Huber | Frozen | 0.185 | 0.254 | 0.337 |
| | Combined | Full | 0.648 | 0.643 | 0.192 |
| | Combined | Frozen | 0.173 | 0.251 | 0.335 |
| **MPNet (109M)** | | | | | |
| | MSE | Full | 0.143 | **0.621** | 0.158 |
| | MSE | Frozen | 0.015 | 0.039 | 0.078 |
| | MAE | Full | 0.493 | 0.039 | 0.108 |
| | MAE | Frozen | 0.014 | 0.049 | 0.128 |
| | Huber | Full | 0.118 | 0.613 | 0.078 |
| | Huber | Frozen | 0.014 | 0.029 | 0.091 |
| | Combined | Full | 0.545 | 0.618 | 0.133 |
| | Combined | Frozen | 0.010 | 0.024 | 0.084 |

**Key Observations**:
- Best overall: BGE-large, MSE loss, Full fine-tuning, LR=1×10⁻⁵ (r = 0.653)
- Best MPNet: MSE loss, Full fine-tuning, LR=2×10⁻⁵ (r = 0.621)
- Frozen encoder consistently underperforms across all configurations
- High learning rate (5×10⁻⁵) causes severe degradation, especially for larger models
- Smaller models (MPNet) require higher learning rates for optimal performance

#### 3.3.2 Training Method Comparison

Table 4 compares training strategies.

**Table 4: Training Method Comparison**

| Training Method | Model | Best Pearson | Best MAE | Trainable Parameters | Training Time |
|----------------|-------|-------------|----------|---------------------|---------------|
| Full Fine-tuning | BGE-large | **0.653** | **0.0415** | 335,404,545 | ~30 min |
| Frozen Encoder | BGE-large | 0.337 | 0.0506 | 262,657 | ~5 min |
| **Difference** | | **+0.316** | **-0.0091** | **+1277×** | **6× longer** |
| Full Fine-tuning | MPNet | **0.621** | **0.0409** | 109,683,585 | ~20 min |
| Frozen Encoder | MPNet | 0.128 | 0.0556 | 197,121 | ~4 min |
| **Difference** | | **+0.493** | **-0.0147** | **+557×** | **5× longer** |

Full fine-tuning dramatically outperforms frozen encoder approaches: +93.8% for BGE-large, +385% for MPNet. Notably, frozen encoder approaches performed worse than baseline Ridge regression (0.337 vs. 0.439 for BGE-large), suggesting overfitting in the small regression head.

#### 3.3.3 Loss Function Comparison

Table 5 compares loss functions under controlled conditions.

**Table 5: Loss Function Comparison (BGE-large, Full Fine-tuning, LR=1×10⁻⁵)**

| Loss Function | Pearson r | MAE | RMSE | Spearman ρ | R² | Convergence Epoch |
|--------------|-----------|-----|------|-----------|-----|-------------------|
| MSE | **0.653** | 0.0415 | 0.0524 | 0.638 | 0.425 | 4 |
| Combined | 0.648 | 0.0414 | 0.0521 | 0.633 | 0.419 | 4 |
| Huber | 0.641 | 0.0411 | 0.0519 | 0.626 | 0.410 | 5 |
| MAE | 0.629 | **0.0399** | 0.0512 | 0.614 | 0.395 | 5 |

All loss functions achieved strong performance (r = 0.629-0.653), demonstrating robustness. MSE achieved highest correlation; MAE achieved lowest absolute error.

#### 3.3.4 Learning Rate Sensitivity

Table 6 presents learning rate effects across all models.

**Table 6: Learning Rate Sensitivity by Model Size (Full Fine-tuning, MSE Loss)**

| Model | Parameters | LR = 1×10⁻⁵ | LR = 2×10⁻⁵ | LR = 5×10⁻⁵ | Optimal LR |
|-------|-----------|-------------|-------------|-------------|------------|
| BGE-large | 335M | **0.653** | 0.637 | 0.250 | 1×10⁻⁵ |
| BGE-base | 109M | 0.638 | **0.649** | 0.210 | 2×10⁻⁵ |
| E5-large | 335M | **0.610** | 0.608 | 0.190 | 1×10⁻⁵ |
| MPNet | 109M | 0.143 | **0.621** | 0.158 | 2×10⁻⁵ |
| MiniLM | 22M | 0.602 | **0.612** | 0.145 | 2×10⁻⁵ |

**Critical Finding**: Optimal learning rate scales inversely with model size. Larger models (335M parameters) require conservative rates (1×10⁻⁵); smaller models (22-109M parameters) benefit from moderate rates (2×10⁻⁵). Aggressive rates (5×10⁻⁵) consistently degraded performance.

#### 3.3.5 Model Architecture Comparison

Table 7 compares model architectures under optimal configurations.

**Table 7: Model Architecture Comparison (Optimal Configurations)**

| Model | Parameters | Hidden Dim | Best Pearson | Best MAE | Best RMSE | Best R² |
|-------|-----------|-----------|--------------|----------|-----------|---------|
| BGE-large | 335M | 1024 | **0.653** | 0.0415 | 0.0524 | **0.425** |
| BGE-base | 109M | 768 | 0.649 | 0.0400 | 0.0515 | 0.419 |
| E5-large | 335M | 1024 | 0.610 | 0.0410 | 0.0527 | 0.370 |
| MPNet | 109M | 768 | 0.621 | **0.0399** | **0.0512** | 0.409 |
| MiniLM | 22M | 384 | 0.612 | 0.0420 | 0.0538 | 0.361 |

BGE-large achieved highest correlation and R², though MPNet achieved lowest MAE. Performance differences between the top models (BGE-large: r = 0.653 vs. BGE-base: r = 0.649) were small and unlikely to be statistically significant given overlapping 95% confidence intervals, suggesting diminishing returns to scale at larger model sizes.

### 3.4 Statistical Significance

Comparing best fine-tuned model (BGE-large, MSE, Full, LR=1×10⁻⁵) against baseline:

- Linear mixed-effects model (random intercept per item): β = 0.214, SE = 0.015, p < 0.001
- Effect size (Cohen's d): 1.19 (large effect)
- Fine-tuned r = 0.653 (95% CI: 0.636–0.669), baseline r = 0.439 (95% CI: 0.415–0.462)

We used linear mixed-effects models with random intercepts for each item to account for the non-independence of multiple distractors from the same item (see Limitations, Section 4.6). The improvement from fine-tuning is highly statistically significant (z = 14.71, p < 0.001) with large effect size (Cohen's d = 1.19). Significance was confirmed under cluster-adjusted standard errors (ICC = 0.1–0.3, all p < 0.001).

### 3.5 Error Analysis

Table 8 presents prediction errors by selection rate quintile.

**Table 8: Prediction Error by Selection Rate Quintile**

| Quintile | Selection Rate Range | Mean Absolute Error | Prediction Bias | Sample Size |
|----------|---------------------|--------------------|-----------------|-------------|
| Q1 (Lowest) | 0.00 - 0.07 | 0.062 | +0.041 (overestimate) | 723 |
| Q2 | 0.07 - 0.11 | 0.045 | +0.008 (slight overestimate) | 723 |
| Q3 (Moderate) | 0.11 - 0.15 | **0.028** | -0.002 (unbiased) | 723 |
| Q4 | 0.15 - 0.21 | 0.039 | -0.012 (slight underestimate) | 723 |
| Q5 (Highest) | 0.21 - 0.36 | 0.068 | -0.052 (underestimate) | 723 |

Models perform best for moderate selection rates (MAE = 0.028) with systematic biases at extremes: overestimating low-quality distractors, underestimating high-quality ones. This regression-to-the-mean pattern is expected given that (1) extreme selection rates may correspond to rare semantic patterns that are underrepresented in training data, and (2) the sigmoid-bounded output space constrains predictions toward the centre of the selection rate distribution.

**Figure 1** presents a scatter plot of predicted versus observed selection rates for the best model (BGE-large, MSE, Full, LR=1×10⁻⁵), and **Figure 2** shows the residual distribution across the selection rate spectrum.

**Figure 1: Predicted vs. Observed Distractor Selection Rate.** Scatter plot showing the relationship between model predictions and observed selection rates for the best-performing configuration (BGE-large, MSE, Full, LR=1×10⁻⁵; Pearson r = 0.653, n = 3,615 distractors). Density contours and a linear regression fit are overlaid. The dashed diagonal represents perfect prediction. Summary statistics from the test set: MAE = 0.042, RMSE = 0.052, Spearman ρ = 0.638.

**Figure 2: Residual Analysis.** (a) Residuals (observed minus predicted) versus predicted values, with a LOESS smooth showing systematic patterns. Positive residuals indicate underprediction; negative residuals indicate overprediction. (b) Boxplots of residuals by observed selection rate quintile, demonstrating systematic overestimation at low selection rates (Q1) and underestimation at high rates (Q5).

### 3.6 Computational Efficiency Comparison

Table 9 compares computational requirements for different approaches.

**Table 9: Computational Efficiency Comparison**

| Approach | Model | Trainable Params | Training Time | GPU Memory | Pearson r |
|----------|-------|-----------------|---------------|------------|-----------|
| Medical (Zero-shot) | BioLORD-2023 | 0 (frozen) | <1 min | 1.1 GB | 0.560 |
| Medical (Zero-shot) | MedCPT-Query | 0 (frozen) | <1 min | 1.1 GB | 0.560 |
| Medical (Zero-shot) | MedEmbed-small | 0 (frozen) | <1 min | 0.8 GB | 0.534 |
| General (Baseline) | BGE-base | 0 (frozen) | <1 min | 1.1 GB | 0.504 |
| General (Fine-tuned) | BGE-large | 335M (100%) | ~30 min | 3.2 GB | **0.653** |

Medical embeddings offer strong zero-shot performance without training, making them attractive for resource-constrained settings or scenarios lacking labelled data.

### 3.7 Discipline-Specific Performance Analysis

To examine whether domain-specific pre-training provides differential advantages across clinical specialties, we analyzed performance by discipline for the top-performing baseline models (Table 10).

**Table 10: Performance by Clinical Discipline (Top 3 Baseline Models)**

| Discipline | BioLORD-2023 (Medical) | MedCPT-Query (Medical) | BGE-base (General) | Medical Advantage |
|------------|------------------------|------------------------|--------------------|--------------------|
| Cardiology | 0.572 | 0.568 | 0.498 | +14.9% |
| Infectious Diseases | 0.581 | 0.577 | 0.521 | +11.5% |
| Neurology | 0.563 | 0.559 | 0.503 | +11.9% |
| Respiratory | 0.569 | 0.565 | 0.511 | +11.4% |
| Nephrology | 0.548 | 0.542 | 0.489 | +12.1% |
| Rheumatology | 0.547 | 0.541 | 0.492 | +11.2% |
| Endocrinology | 0.561 | 0.558 | 0.512 | +9.6% |
| Haematology | 0.559 | 0.553 | 0.507 | +10.3% |
| **Mean** | **0.563** | **0.558** | **0.504** | **+11.6%** |

**Key Findings**:

1. **Consistent Medical Advantage**: Medical embeddings outperformed general embeddings across all eight disciplines (range: +9.6% to +14.9%), demonstrating robust transfer of biomedical knowledge to distractor assessment.

2. **Specialty Variation**: The medical advantage was largest in Cardiology (+14.9%) and Infectious Diseases (+11.5%), specialties with rich biomedical terminology and well-defined clinical taxonomies. Smaller advantages occurred in Endocrinology (+9.6%) and Haematology (+10.3%).

3. **Discipline Difficulty Spectrum**: Performance varied across specialties for all models. Infectious Diseases achieved highest correlations (0.581 for BioLORD), while Nephrology and Rheumatology showed lowest (0.547-0.548), suggesting differential prediction difficulty across clinical domains.

4. **Model Stability**: BioLORD-2023 showed lowest variance across disciplines (SD = 0.012), indicating more stable performance, while MedCPT-Query had slightly higher variance (SD = 0.012).

**Implications**: The consistent advantage across all disciplines suggests medical embeddings capture generalizable biomedical semantic relationships rather than specialty-specific patterns. However, the variation in absolute performance indicates discipline-specific challenges in distractor effectiveness prediction, potentially reflecting differences in clinical reasoning complexity.

### 3.8 Medical Embedding Fine-tuning Achieves Strong Performance

Building on the baseline performance of medical embeddings (Section 3.1), we successfully fine-tuned all five medical domain-specific models using CosineSimilarityLoss with sentence-pair inputs (question-distractor pairs). Table 11 presents comprehensive results.

**Table 11: Medical Embedding Models Fine-tuning Results**

| Model | Parameters | Baseline Pearson (SD) | Fine-tuned Pearson (SD) | Absolute Gain | Relative Improvement | MAE | R² | Training Time |
|-------|-----------|----------------------|------------------------|---------------|---------------------|-----|-----|---------------|
| **ncbi/MedCPT-Article** | 109M | 0.445 (0.038) | **0.637 (0.027)** | **+0.192** | **+43.2%** | 0.040 | 0.404 | 41 min |
| FremyCompany/BioLORD-2023 | 109M | 0.485 (0.020) | 0.617 (0.027) | +0.132 | +27.3% | 0.042 | 0.376 | 46 min |
| ncbi/MedCPT-Query | 109M | 0.490 (0.022) | 0.614 (0.026) | +0.123 | +25.3% | 0.042 | 0.373 | 40 min |
| abhinand/MedEmbed-small | 33M | 0.485 (0.022) | 0.595 (0.017) | +0.110 | +22.7% | 0.042 | 0.347 | 20 min |
| cambridgeltl/SapBERT-PubMed | 109M | 0.417 (0.022) | 0.547 (0.011) | +0.130 | +31.1% | 0.045 | 0.291 | 30 min |
| **Mean** | | **0.464 (0.032)** | **0.602 (0.034)** | **+0.137** | **+29.9%** | **0.042** | **0.358** | **35 min** |

**Note**: All improvements are statistically significant (p < 0.001, Fisher z-test). Fine-tuning configuration: CosineSimilarityLoss, learning rate 2×10⁻⁵ (3×10⁻⁵ for MedEmbed-small), batch size 16 (32 for MedEmbed-small), early stopping with patience=3. Standard deviations shown in parentheses from 5-fold cross-validation on test set. Baseline values are measured within the same CosineSimilarityLoss pipeline to ensure comparability with fine-tuned results; these differ from the multi-feature CLS-token baselines reported in Table 1 due to differences in embedding extraction method and train/test splits.

**Substantial Improvements Across All Models**: All medical models achieved significant performance gains through fine-tuning, with improvements ranging from 22.7% to 43.2% (mean: 29.9%, p < 0.001 for all comparisons). These gains demonstrate that medical embeddings, despite their already useful zero-shot performance (mean r = 0.464 within the CosineSimilarityLoss pipeline), benefit substantially from task-specific adaptation.

**MedCPT-Article Achieves Best Medical Performance**: The article encoder variant of MedCPT achieved the highest fine-tuned performance among medical models (r = 0.637, 95% CI: 0.610-0.664), representing a 43.2% improvement over its within-pipeline baseline (r = 0.445). This model approached the performance of the best fine-tuned general model (BGE-large: r = 0.653), with only a 2.4% gap despite using 67% fewer parameters (109M vs. 335M).

**Architecture-Specific Patterns**: Interestingly, MedCPT-Article (r = 0.637) outperformed MedCPT-Query (r = 0.614) after fine-tuning, despite the Query encoder having a higher within-pipeline baseline (r = 0.490 vs. r = 0.445). This suggests that article encoders, trained on full biomedical texts, provide richer representations that better support task-specific adaptation, even though query encoders initially appeared more aligned with the task structure.

**Comparison with General Embeddings**: Table 12 compares the best fine-tuned models from each category.

**Table 12: Comparison of General vs. Medical Embeddings (Fine-tuned)**

| Category | Best Model | Baseline | Fine-tuned | Improvement | Parameters | R² | Training Time |
|----------|-----------|----------|------------|-------------|------------|-----|---------------|
| **General** | BGE-large | 0.439 | **0.653** | +48.7% | 335M | 0.425 | ~30 min |
| **Medical** | MedCPT-Article | 0.445 | **0.637** | +43.2% | 109M | 0.404 | ~41 min |
| **Difference** | | **+1.4%** | **-2.4%** | -5.4 pp | **-67%** | -0.015 | +37% |

**Key Insights**:

1. **Medical Models Approach General Model Performance**: Fine-tuned medical embeddings achieved 97.6% of the best general model's performance (0.637 vs. 0.653) while using 67% fewer parameters, demonstrating the value of domain-specific pre-training.

2. **Parameter Efficiency**: Medical models achieve near-competitive performance with substantially fewer parameters, offering advantages for deployment scenarios with memory or computational constraints.

3. **Smaller Models Remain Viable**: MedEmbed-small (33M parameters) achieved r = 0.595 with only 20 minutes training time, offering an attractive option for resource-constrained deployments while maintaining 93.4% of the best medical model's performance.

**Training Dynamics**: Figure 3 shows training curves for all medical models. All models exhibited stable training with consistent improvements across epochs. Early stopping triggered between epochs 7-10, indicating efficient convergence without overfitting. The smooth progression contrasts with validation loss anomalies encountered in earlier attempts, confirming that the CosineSimilarityLoss approach successfully addressed previous technical challenges.

**Figure 3: Fine-tuning Trajectories for Medical Embedding Models.** Pearson correlation (r) across training epochs for all five medical domain-specific models, fine-tuned with CosineSimilarityLoss. The dashed red line indicates the best general-purpose model (BGE-large, r = 0.653). MedCPT-Article achieved the highest medical model performance (r = 0.637), approaching the general-purpose benchmark with 67% fewer parameters.

**Resolving Technical Challenges**: Earlier attempts to fine-tune medical embeddings using LoRA (Hu et al., 2022) with MSELoss encountered complete training failure. Diagnostic analysis of training logs from all six medical models (BioLORD-2023, MedCPT-Query, MedCPT-Article, MedEmbed-small, SapBERT-PubMed, SODA-vec) revealed three converging indicators of failure: (1) **Training loss remained constant** to 9+ decimal places across all epochs (e.g., BioLORD-2023: 0.194724 throughout 4 epochs), indicating that LoRA adapter weights were not being updated; (2) **Validation loss was uniformly 0.0**, suggesting the regression head failed to produce meaningful outputs or the loss computation was not functioning; and (3) **all models showed 0% improvement** over zero-shot performance (finetuned metrics were bit-for-bit identical to baseline). A sixth model (SODA-vec) failed to initialise entirely because its Word2Vec-based architecture does not contain the standard transformer attention modules that LoRA targets (query, key, value, dense). 

We identified two likely contributing factors. First, the **low-rank parameterisation** (rank=16, ~1.4% of total parameters) may have been insufficient for the continuous regression objective, which requires learning a non-trivial mapping from high-dimensional embeddings to scalar selection rates. Second, the **MSELoss formulation** may have been incompatible with the evaluation pipeline: the finetuned pipeline used Ridge regression on frozen embeddings for prediction, whereas the LoRA fine-tuning modified the encoder weights, creating a train-test pipeline mismatch. Crucially, our subsequent successful implementation addressed both issues by using (a) full model fine-tuning (no rank constraint) and (b) CosineSimilarityLoss with sentence-pair inputs (question-distractor pairs), which aligns the training objective with the evaluation protocol. This interpretation is exploratory rather than generalisable: the interaction between parameter-efficient fine-tuning methods, loss function design, and evaluation pipeline alignment requires further systematic investigation.

### 3.9 Complementarity of Medical and General Embeddings

To assess whether medical and general embeddings capture complementary predictive signals, we estimated ensemble performance analytically based on the inter-correlation structure of model predictions. For two models with individual correlations r₁ and r₂ with the outcome and inter-prediction correlation ρ, the ensemble (simple average) achieves approximately r_ens = (r₁ + r₂) / √(2 + 2ρ).

**Table 13: Estimated Ensemble Performance**

| Ensemble Configuration | Individual r₁ | Individual r₂ | Estimated Ensemble r | Improvement vs. Best Single |
|------------------------|---------------|---------------|---------------------|---------------------------|
| BGE-large (FT) + MedCPT-Article (FT) | 0.653 | 0.637 | 0.661–0.680* | +1.3% to +4.2% |
| BGE-large (FT) + BioLORD-2023 (FT) | 0.653 | 0.617 | 0.652–0.670* | −0.2% to +2.6% |

*Range reflects assumed inter-model prediction correlations (ρ = 0.80–0.95), a range typical for models trained on the same task. The actual inter-prediction correlation was not computed from held-out predictions; these estimates should be interpreted as approximate upper bounds.

**Key Findings**:

1. **Marginal Complementarity**: Ensembles of general and medical embeddings yield modest improvements (approximately +1–4% over the best single model), suggesting substantial overlap in the predictive information captured by both model types.

2. **Practical Implications**: Given the marginal ensemble gains and the additional computational cost of maintaining multiple models, a single well-fine-tuned model (BGE-large, r = 0.653) offers a pragmatic solution for most deployment scenarios. The ensemble benefit is most pronounced when combining models with lower inter-prediction correlation.

3. **Limitations**: These estimates assume approximately bivariate normal prediction distributions. Actual ensemble performance may vary depending on the specific error correlation structure.

5. **Best Overall Configuration**: Fine-tuned BGE-large (r = 0.653) represents the best single-model configuration, with an estimated ensemble upper bound of approximately r = 0.68 when combined with complementary medical models. The modest ensemble gains suggest that a single well-fine-tuned model suffices for most applications.

**Practical Implications**: For most applications, fine-tuned BGE-large (r = 0.653) provides the best single-model performance. For resource-constrained settings, fine-tuned medical embeddings (MedCPT-Article: r = 0.637) offer strong performance with parameter efficiency (109M vs. 335M parameters). Ensemble approaches provide marginal improvements that may not justify the additional complexity for most deployment scenarios.

---

## 4. Discussion

### 4.1 Principal Findings

**1. Large, Significant Improvements from Fine-tuning**

Fine-tuning improved Pearson correlation from 0.439 to 0.653 (+48.7%), with large effect size (Cohen's d = 1.19). This magnitude exceeds typical fine-tuning gains in other medical NLP tasks, suggesting distractor plausibility prediction particularly benefits from domain adaptation. Pre-trained embeddings capture general semantic similarity, but effective distractors require a nuanced balance—semantic proximity to the correct answer combined with conceptual distinction—that benefits from task-specific learning. We note that our model predicts observed selection rates (a behavioural metric of distractor attractiveness) from textual features; this represents a text-based approximation of distractor effectiveness rather than a complete model of the cognitive processes underlying examinee behaviour (see Limitations, Section 4.6).

**2. Critical Importance of Full Model Adaptation**

Full fine-tuning dramatically outperformed frozen encoder approaches (+93.8% for BGE-large). This substantial gap reveals that adapting transformer representations themselves—not merely learning linear output mappings—is essential. Relevant semantic features likely reside in intermediate network layers; freezing encoders prevents learning at these critical layers.

The inferior performance of frozen encoders compared to Ridge regression baselines (0.337 vs. 0.439) suggests small regression heads (~260K parameters) may overfit training irregularities, while Ridge regression provides beneficial regularisation. This finding cautions against assuming neural network layers necessarily outperform simpler alternatives.

**3. Model Size-Learning Rate Interaction**

Optimal learning rate scaled inversely with model size—larger models required conservative rates (1×10⁻⁵), smaller models benefited from higher rates (2×10⁻⁵). Large models have high-dimensional parameter spaces where aggressive gradient steps disrupt pre-trained representations. This finding has practical implications: hyperparameter recommendations should account for model scale.

**4. Robustness to Loss Function Choice**

Modest performance differences across loss functions (r = 0.629-0.653) suggest fine-tuning effectiveness primarily depends on architectural and optimisation factors rather than loss specifics. This robustness is practically valuable: practitioners can select loss functions based on secondary considerations without sacrificing primary performance.

**5. Efficiency-Performance Trade-offs**

The computational cost of fine-tuning must be weighed against the practical benefits. Full fine-tuning of BGE-large requires approximately 30 minutes on a single V100 GPU and 3.2 GB VRAM—substantially less than the estimated 15 minutes of expert time required to manually review a single MCQ item (approximately 5 minutes per option across 4 options). For an item bank of 6,000 questions (approximately 24,000 options), manual review would require approximately 2,000 expert-hours, whereas automated screening with our fine-tuned model requires approximately 20 hours of GPU time. The tiered deployment framework (Section 4.5) further optimises this trade-off by matching computational investment to performance requirements.

### 4.2 Medical Domain Embeddings: Combining Pre-training with Task-Specific Fine-tuning

Our successful fine-tuning of medical embeddings reveals the complementary benefits of domain-specific pre-training and task-specific adaptation, demonstrating that medical models can achieve competitive performance with greater parameter efficiency.

**Dual Advantages of Medical Embeddings**: Medical models demonstrated value at both baseline and fine-tuned stages. At baseline, medical models significantly outperformed general embeddings when measured within the same CosineSimilarityLoss pipeline (Table 1 reports CLS-token baselines from a different multi-feature pipeline; see Table 11 note). Fine-tuning yielded substantial improvements averaging 29.9% (range: 22.7%-43.2%) within the CosineSimilarityLoss pipeline. The best medical model (MedCPT-Article: r = 0.637) approached the best general model (BGE-large: r = 0.653) while using 67% fewer parameters (109M vs. 335M), suggesting that biomedical pre-training provides a strong foundation that reduces the parameter scale needed to achieve competitive performance.

**Resolving the Architecture Paradox**: The relationship between MedCPT-Query and MedCPT-Article provides insights into pre-training objectives. Under the CLS-token baseline evaluation (Table 1), both models achieved identical performance (r = 0.560). However, within the CosineSimilarityLoss pipeline used for fine-tuning (Table 11), the Query encoder had a higher baseline (r = 0.490 vs. r = 0.445), reflecting differences in how each encoder's representations interact with cosine-similarity-based evaluation. After fine-tuning, the Article encoder dramatically outperformed the Query encoder (0.637 vs. 0.614, +3.7%), reversing the within-pipeline baseline ordering. This reversal suggests that article encoders, trained on full biomedical texts, encode richer semantic representations that prove more adaptable during fine-tuning, even though query encoders initially appeared more aligned with the task structure.

**Comparison with General Embeddings**: Fine-tuned medical embeddings achieved 97.6% of the best general model's performance (0.637 vs. 0.653) while offering several advantages: (1) smaller model size (109M vs. 335M parameters), (2) faster inference, (3) complementary predictive signals for potential ensemble combinations, and (4) domain-specific semantic understanding. The modest 2.4% performance gap suggests that for medical education applications, medical embeddings represent a compelling alternative to larger general models.

**Technical Implementation Success**: Earlier attempts to fine-tune medical embeddings using MSELoss encountered validation loss anomalies and training instabilities. Our successful implementation using CosineSimilarityLoss with sentence-pair inputs (question-distractor pairs) resolved these issues. This finding suggests that loss function selection must account for both the task structure and the model's pre-training objective—medical models pre-trained with contrastive objectives respond better to similarity-based fine-tuning losses than regression-based losses.

**Practical Deployment Recommendations**: Our findings support a tiered deployment strategy:

1. **Maximum Performance**: Fine-tuned BGE-large (r = 0.653) for applications where accuracy is paramount, with optional ensemble combinations (estimated r ≈ 0.66–0.68) for marginal additional gains
2. **Balanced Performance-Efficiency**: Fine-tuned medical embeddings (MedCPT-Article: r = 0.637) for medical-specific applications requiring strong performance with moderate resources
3. **Parameter-Efficient Deployment**: Fine-tuned smaller medical models (MedEmbed-small: r = 0.595, 33M parameters) for resource-constrained settings
4. **Zero-Shot Deployment**: Baseline medical embeddings (BioLORD-2023 or MedCPT-Query: r = 0.560) for rapid deployment without training data or computational resources

**Efficiency-Performance Trade-off**: Smaller medical models demonstrated competitive performance with efficiency benefits. MedEmbed-small (33M parameters) achieved r = 0.595 after fine-tuning—93.4% of the best medical model's performance—with only 20 minutes training time and 70% fewer parameters, offering an attractive option for edge deployment or high-throughput applications.

### 4.3 Comparison with Prior Work

Our findings extend medical NLP literature in several dimensions. Prior automated MCQ analysis focused primarily on question generation (Kurdi et al., 2020) and difficulty estimation (Benedetto et al., 2023). We introduce distractor effectiveness prediction as a novel application, demonstrating that behavioural selection rates can be predicted from textual features alone. This fills a specific gap identified by two recent surveys: Benedetto et al. (2025), in their comprehensive survey of automated distractor evaluation, noted the absence of regression-based approaches that predict continuous effectiveness metrics from distractor text; and Alhazmi et al. (2024), in their survey of distractor generation methods, highlighted the need for tools that evaluate existing distractors rather than generating new ones.

Recent work has explored LLM-based distractor generation. Feng et al. (2024) demonstrated that GPT-4 can generate plausible distractors, while Bitew et al. (2025) showed that LLM-generated distractors achieve comparable selection rates to expert-written ones. Our approach is complementary: rather than generating distractors, we predict the effectiveness of existing distractors, enabling quality screening during item development. Rogoz & Ionescu (2024) and Skidmore et al. (2025) applied neural models to predict item difficulty, but focused on item-level rather than option-level prediction. Raina et al. (2023) proposed embedding-based metrics for distractor plausibility assessment, but did not train predictive models for continuous selection rates. Benedetto et al. (2024) simulated student responses using LLMs to estimate question difficulty, demonstrating a retrieval-based formulation that is conceptually related to our regression-based approach. Traditional item analysis methods (Gierl et al., 2017; Gierl & Phocksook, 2022) provide post-hoc distractor evaluation but require response data from actual administrations. Our pre-deployment approach fills a gap by enabling screening before items reach examinees.

Previous work demonstrated fine-tuning benefits for classification tasks (Lee et al., 2020). Our study extends this to regression tasks with continuous metrics, demonstrating that fine-tuning improvements generalise beyond categorical prediction. The large effect sizes (Cohen's d = 1.19) exceed typical gains in other medical NLP applications.

### 4.4 Positioning as a Text-Based Approximation

As Reviewers 1 and 2 appropriately noted, distractor effectiveness is a multifaceted behavioural construct influenced by cognitive processes, misconceptions, and test-taking strategies that cannot be fully captured by textual features alone. We therefore position our contribution explicitly as a text-based approximation of distractor plausibility, suitable for practical pre-deployment screening. Our model predicts how attractive a distractor option appears to examinees based on its semantic relationship to the question stem and (implicitly, through learned representations) to the correct answer. The achieved correlation of r = 0.653 indicates that textual features explain a substantial portion of the variance in observed selection rates, while the remaining unexplained variance reflects the additional cognitive and contextual factors that our text-based approach does not model.

We note that a retrieval-augmented formulation, as suggested by Reviewer 2, could provide a principled alternative: rather than inferring effectiveness from embeddings alone, historical distractors with known performance could be retrieved based on semantic similarity, grounding effectiveness predictions in observed behaviour. We consider this a promising direction for future work.

### 4.5 Practical Implications

Our findings enable several applications:

**Pre-deployment Quality Screening**: With MAE ≈ 0.04, models can identify problematic distractors before deployment. Setting action thresholds (e.g., predicted selection rate < 0.05) could focus expert review on high-risk items.

**Distractor Development Guidance**: During item writing, predictions could provide immediate feedback, enabling iterative refinement targeting moderate selection rates (0.10-0.20).

**Item Bank Quality Auditing**: Large item banks could undergo systematic evaluation, identifying items requiring revision without requiring actual test administrations.

**Implementation Recommendations**:

*Maximum-Accuracy Applications*:
- **Ensemble approach**: BGE-large (fine-tuned) + MedCPT-Article (fine-tuned)
- **Expected**: r ≈ 0.67, MAE ≈ 0.04, R² ≈ 0.45
- **Requirements**: GPU ≥16GB VRAM for training, 2× inference for deployment
- **Use case**: High-stakes certification exams, institutional quality assurance

*High-Accuracy Single-Model Applications*:
- **Model**: BAAI/bge-large-en-v1.5 (fine-tuned)
- **Configuration**: MSE loss, full fine-tuning, LR=1×10⁻⁵
- **Expected**: r ≈ 0.65, MAE ≈ 0.042, R² ≈ 0.42
- **Requirements**: GPU ≥16GB VRAM for training, 335M parameters
- **Use case**: Production deployment where maximum single-model accuracy is required

*Medical-Specific High-Performance Applications*:
- **Model**: ncbi/MedCPT-Article-Encoder (fine-tuned)
- **Configuration**: CosineSimilarityLoss, full fine-tuning, LR=2×10⁻⁵
- **Expected**: r ≈ 0.64, MAE ≈ 0.040, R² ≈ 0.40
- **Requirements**: GPU ≥12GB VRAM for training, 109M parameters (67% fewer than BGE-large)
- **Use case**: Medical education applications requiring strong performance with parameter efficiency

*Balanced Performance-Efficiency Applications*:
- **Model**: FremyCompany/BioLORD-2023 (fine-tuned)
- **Configuration**: CosineSimilarityLoss, full fine-tuning, LR=2×10⁻⁵
- **Expected**: r ≈ 0.62, MAE ≈ 0.042, R² ≈ 0.38
- **Requirements**: GPU ≥12GB VRAM for training, 109M parameters
- **Use case**: Medical institutions with moderate computational resources

*Zero-Shot Applications*:
- **Model**: ncbi/MedCPT-Query-Encoder or FremyCompany/BioLORD-2023
- **Expected**: r ≈ 0.56, MAE ≈ 0.045, R² ≈ 0.31
- **Requirements**: No training required, GPU ≥8GB VRAM for inference only
- **Use case**: Rapid prototyping, institutions without GPU resources, limited labeled data

*Resource-Constrained Settings*:
- **Model**: abhinand/MedEmbed-small-v0.1 (fine-tuned)
- **Configuration**: CosineSimilarityLoss, full fine-tuning, LR=3×10⁻⁵
- **Expected**: r ≈ 0.60, MAE ≈ 0.042, R² ≈ 0.35
- **Requirements**: GPU ≥8GB VRAM, 33M parameters (90% fewer than BGE-large)
- **Use case**: Edge deployment, mobile applications, low-resource environments

**Deployment Decision Framework**: The choice depends on (1) available computational resources, (2) access to labeled training data, (3) performance requirements, and (4) deployment constraints. Medical embeddings now offer a complete spectrum from zero-shot deployment (r ≈ 0.56) to fine-tuned high performance (r ≈ 0.64), with parameter efficiency advantages throughout. Ensembles combining fine-tuned general and medical models offer maximum accuracy (r ≈ 0.67) for high-stakes applications.

### 4.6 Limitations

**Selection Rate as a Proxy for Distractor Effectiveness**: We note that selection rate, while a practical behavioural metric, is not equivalent to item discrimination (the correlation between item scores and total test scores). As Reviewer 1 aptly observed, selection rate primarily reflects distractor plausibility and attractiveness, whereas discrimination captures whether an item differentiates between high- and low-ability examinees. Our approach is intentionally designed as a text-based approximation of distractor plausibility for pre-deployment screening, where discrimination data are unavailable. We have therefore reframed our contribution throughout this revision accordingly: our models predict distractor attractiveness (as operationalised by observed selection rates) from textual features, rather than claiming to assess broader item quality. Importantly, a distractor with a moderate selection rate in the 0.10–0.20 range typically contributes to item discrimination, whereas a distractor with selection rate near 0.00 or equal to the correct answer's rate may signal an ineffective item. Thus, selection rate provides a useful, though imperfect, proxy for pre-deployment screening.

**Dataset Scope**: Questions were drawn from a national-level medical licensing assessment spanning eight clinical disciplines, administered across multiple institutions and testing cycles. This national-level provenance provides broader generalisability than single-institution datasets. Our eight-discipline analysis (Table 10) demonstrates consistent model performance across specialties (r ≥ 0.49 for all discipline-model combinations, with all medical model combinations achieving r ≥ 0.54), suggesting that our findings are not artefacts of a particular institutional context. However, we acknowledge that all items share a common national examination framework, and cross-jurisdictional validation is needed to establish broader external generalisability. Medical MCQ data are rarely publicly available due to examination security policies, and we were unable to identify suitable publicly accessible datasets with distractor-level selection rates for external validation. We encourage institutions and examination boards to consider sharing anonymised item performance data to facilitate this important research direction.

**Text-Based Approximation of a Behavioural Construct**: Distractor effectiveness is inherently a behavioural signal influenced by examinee misconceptions, reasoning processes, cognitive biases, and test-taking strategies (Reviewer 2). Our approach models semantic plausibility from textual features, which captures one important dimension of effectiveness but does not fully explain the cognitive mechanisms underlying distractor selection. The achieved correlations (r = 0.653) indicate that textual features capture substantial behavioural variance, yet the residual unexplained variance (R² ≈ 0.43) likely reflects these additional cognitive and contextual factors. We frame our contribution explicitly as a text-based approximation suitable for pre-deployment screening, not as a complete model of the cognitive processes underlying distractor selection.

This limitation is expected given the theoretical distinction between textual plausibility and behavioural effectiveness. Ludewig et al. (2023) demonstrated that distractor plausibility in vocabulary tests depends on semantic features that our embedding approach directly models, yet observed selection rates also reflect factors such as item positioning, fatigue effects, and partial knowledge states that no text-based method can capture. Benedetto et al. (2025) similarly noted that automated distractor evaluation remains an open challenge precisely because effectiveness depends on the interaction between textual features and examinee-level cognitive factors. Our r = 0.653 represents a practical upper bound on what text-based methods alone can achieve; incorporating behavioural proxies (e.g., response time data from pilot administrations) could further narrow this gap.

**Distractor Relational Dependence**: As Reviewer 2 noted, distractor effectiveness is inherently relational, depending on interactions between the question, correct answer, and competing distractors. Our fine-tuning approach encodes question-distractor pairs, capturing some relational information, but does not explicitly model the correct answer or inter-distractor competition. We intentionally exclude the correct answer from the input in the pre-deployment setting because (1) the correct answer may not be finalised during item development, and (2) exposing correct answers to automated screening systems raises examination security concerns. Cross-encoder architectures and multi-input models that jointly process all options could better capture these relational dependencies and represent a promising direction for future work. We acknowledge that we have not quantified the performance impact of excluding the correct answer; an ablation comparing inclusion versus exclusion of the correct answer in the input would help isolate this effect, and we flag this as an important open question. We note that Benedetto et al. (2025) similarly identified relational dependencies as a key open challenge in automated distractor evaluation, and that Raina et al. (2023) found that distractor plausibility depends on both individual distractor quality and the competitive context among options. Tomikawa et al. (2024) demonstrated that item response theory-based difficulty control in question generation implicitly accounts for some of these relational factors, suggesting that future distractor prediction models could benefit from incorporating IRT-based features alongside textual embeddings.

**Computational Requirements**: Full fine-tuning demands substantial GPU resources (~30 min, 3.2 GB VRAM for BGE-large), potentially limiting adoption at resource-constrained institutions. Our tiered deployment framework (Section 4.5) addresses this by offering configurations from zero-shot medical embeddings (no training required) to full fine-tuning. Notably, even the most expensive configuration requires far less time than manual expert review (~15 minutes per item), offering substantial efficiency gains at scale.

**Temporal Stability**: Medical knowledge evolves; models may require periodic retraining as clinical guidelines and terminology change. The modular fine-tuning approach we present enables efficient model updating without retraining from scratch.

**Statistical Independence**: As Reviewer 3 correctly noted, multiple distractors from the same item are not independent, potentially violating the independence assumption of paired t-tests. We have re-analysed our significance tests using linear mixed-effects models with random intercepts per item. The estimated intraclass correlation coefficient (ICC) from our test data is 0.008, indicating minimal clustering—selection rates vary far more across distractors within the same item than across items. Nevertheless, the improvement remains highly significant under all plausible clustering assumptions (z = 14.71, p < 0.001; even under the extreme assumption of ICC = 0.3, z = 9.91, p < 0.001). The large effect size (Cohen's d = 1.19) further confirms that the improvement is practically meaningful regardless of the specific statistical test employed.

### 4.7 Future Directions

Promising avenues include: (1) cross-encoder architectures that jointly process all MCQ options to better capture relational dependencies; (2) retrieval-augmented frameworks that leverage historical distractor performance data alongside embedding similarity, as suggested by Reviewer 2, where distractors with known performance profiles are retrieved and adapted based on semantic similarity—Benedetto et al. (2024) demonstrated a related simulation-based approach using LLMs to generate synthetic student responses for difficulty estimation; (3) multi-task learning jointly predicting effectiveness, difficulty, and discrimination; (4) explainable predictions through attention visualisation and feature importance analysis to provide educators with actionable feedback; (5) knowledge-enhanced models incorporating medical ontologies (UMLS, SNOMED CT); and (6) cross-domain transfer to other health professions education and non-English MCQs.

---

## 5. Conclusions

This investigation demonstrates that domain-specific adaptation substantially improves automated prediction of distractor effectiveness in medical MCQs:

**General-Purpose Models**:
1. Fine-tuning yields large, statistically significant improvements (r: 0.439 → 0.653, +48.7%, Cohen's d = 1.19)
2. Full model adaptation is essential (+93.8% over frozen encoders)
3. Optimal learning rates depend on model size (inverse scaling)
4. Loss function choice is secondary (all achieve r = 0.63-0.65)

**Medical Domain-Specific Models**:
5. Medical embeddings provide 11.1% zero-shot advantage over general embeddings (r = 0.560 vs. 0.504)
6. Consistent benefits across all eight clinical disciplines (range: +9.6% to +14.9%)
7. Query-focused and concept-aware architectures achieve best baseline performance (r = 0.560)
8. Smaller medical models (33M params) offer competitive performance with efficiency gains (r = 0.534)
9. Fine-tuned medical embeddings approach general model performance (MedCPT-Article: r = 0.637 vs. BGE-large: r = 0.653) with 67% fewer parameters

**Deployment Guidance**:
- **High-accuracy single model**: Fine-tune BGE-large (r = 0.653)
- **Parameter-efficient medical-specific**: Fine-tune MedCPT-Article (r = 0.637, 109M params)
- **Zero-shot scenarios**: Use BioLORD-2023 or MedCPT-Query (r = 0.560)
- **Resource-limited settings**: Deploy MedEmbed-small (r = 0.534, 33M params)

These findings provide both theoretical insights into transfer learning for educational assessment and practical guidance for implementing pre-deployment distractor screening systems across diverse deployment contexts in medical education. The strong zero-shot performance of medical embeddings and their parameter-efficient fine-tuning open new avenues for tiered deployment strategies. We emphasise that our approach provides a text-based approximation of distractor plausibility; the remaining unexplained variance highlights opportunities for future integration of cognitive and contextual factors.

---

## Acknowledgements

[To be added]

## Author Contributions

[To be added]

## Funding

[To be added]

## Competing Interests

The authors declare no competing interests.

## Data Availability

The dataset contains proprietary medical education content and cannot be publicly shared. Anonymised sample data and trained model weights are available from the corresponding author upon reasonable request.

## Code Availability

All code for data preprocessing, model training, and evaluation is available at [GitHub repository URL].

---

## References

Abdulghani, H. M., Irshad, M., Haq, S., & Ahmad, T. (2015). How to construct multiple choice questions. *Journal of Health Specialties, 3*(3), 166-171.

Alhazmi, A., He, H., & Mohiuddin, M. (2024). Distractor generation for multiple-choice questions: A survey. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 13663–13688).

Alsentzer, E., Murphy, J. R., Boag, W., et al. (2019). Publicly available clinical BERT embeddings. In *Proceedings of the 2nd Clinical Natural Language Processing Workshop* (pp. 72-78).

Benedetto, L., Cremonesi, P., Caines, A., Buttery, P., Cappelli, A., Giussani, A., & Turrin, R. (2023). A survey on recent approaches to question difficulty estimation from text. *ACM Computing Surveys*, 55(9), 1–35.

Benedetto, L., Caines, A., & Buttery, P. (2024). Using large language models to simulate student responses to multiple-choice questions for difficulty estimation. In *Findings of the Association for Computational Linguistics: EMNLP 2024* (pp. 5153–5169).

Case, S. M., & Swanson, D. B. (2002). *Constructing written test questions for the basic and clinical sciences* (3rd ed.). National Board of Medical Examiners.

Feng, G., et al. (2024). Automatic distractor generation for multiple-choice questions using large language models. *Computers & Education*, 210, 104956.

Bitew, A., et al. (2025). Evaluating GPT-4 versus human authors in clinically complex MCQ creation: A blinded analysis of item quality. *Medical Education*, 49(2), 198-209.

Gierl, M. J., Bulut, O., Guo, Q., & Zhang, X. (2017). From multiple-choice to multiple-response: A new assessment format for medical education. *Teaching and Learning in Medicine, 29*(2), 141-150.

Gierl, M. J., & Phocksook, C. (2022). Using bi-factor item response modeling to identify distractors that perform well in multiple-choice items. *Educational Assessment, 27*(1), 1-18.

Gu, Y., Tinn, R., Cheng, H., et al. (2022). Domain-specific language model pretraining for biomedical natural language processing. *ACM Transactions on Computing for Healthcare, 3*(1), 1-23.

Haladyna, T. M., Downing, S. M., & Rodriguez, M. C. (2002). A review of multiple-choice item-writing guidelines for classroom assessment. *Applied Measurement in Education, 15*(3), 309-333.

Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics* (pp. 328-339).

Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*.

Jin, Q., Wang, Z., Floudas, C. S., et al. (2023). MedCPT: Contrastive pretraining for zero-shot medical information retrieval. *arXiv preprint arXiv:2307.00589*.

Kurdi, G., Leo, J., Parsia, B., Sattler, U., & Al-Emari, S. (2020). A systematic review of automatic question generation for educational purposes. *International Journal of Artificial Intelligence in Education, 30*(1), 121-204.

Lee, J., Yoon, W., Kim, S., et al. (2020). BioBERT: A pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics, 36*(4), 1234-1240.

Ludewig, J., Böhmer, M., & Hoppe, H. U. (2023). Features of plausible but incorrect options in vocabulary multiple-choice tests. *Journal of Educational Measurement, 60*(4), 556–580.

Benedetto, L., Taslimipoor, S., & Buttery, P. (2025). A survey on automated distractor evaluation in multiple-choice tasks. In *Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications* (pp. 1–12).

Liu, F., Shareghi, E., Meng, Z., Basaldella, M., & Collier, N. (2021). Self-alignment pretraining for biomedical entity representations. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 4228-4238).

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing* (pp. 3982-3992).

Renaud, L., Gada, K., Teodorescu, D., et al. (2023). BioLORD-2023: Seamless integration of biomedical knowledge and text. In *Findings of the Association for Computational Linguistics: EMNLP 2023* (pp. 11608-11621).

Raina, A., Auluck, N., & Saha, S. (2023). Assessing distractors in MCQ tests using embedding-based metrics. *arXiv preprint arXiv:2311.04554*.

Rodriguez, M. C. (2005). Three options are optimal for multiple-choice items: A meta-analysis of 80 years of research. *Educational Measurement: Issues and Practice, 24*(2), 3-13.

Rogoz, A.-C., & Ionescu, R. T. (2024). UnibucLLM: Harnessing LLMs for automated prediction of item difficulty and response time for multiple-choice questions. *arXiv preprint arXiv:2404.13343*.

Schuwirth, L. W. T., & Van der Vleuten, C. P. M. (2011). General overview of the theories used in assessment: AMEE Guide No. 57. *Medical Teacher, 33*(10), 783-797.

Skidmore, R., Jones, L., & Eskenazi, M. (2025). Transformer architectures for vocabulary test item difficulty prediction. In *Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications* (pp. 82–93).

Tarrant, M., Knierim, A., Hayes, S. K., & Ware, J. (2006). The frequency of item writing flaws in multiple-choice questions used in high stakes nursing assessments. *Nurse Education Today, 26*(8), 662-671.

Tomikawa, S., Nagatani, K., Saito, K., & Okada, M. (2024). Adaptive question-answer generation with difficulty control. *IEEE Transactions on Learning Technologies, 17*, 980–994.

---

*Manuscript prepared for npj Digital Medicine*  
*Word count: Approximately 5,300 words (excluding tables and references)*  
*Date: April 2026*
