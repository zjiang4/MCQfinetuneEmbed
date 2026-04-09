# Response to Reviewers

**Manuscript ID:** b6f32430-e205-4ad2-8932-f899b055b2e3  
**Title:** Transfer Learning for Automated Distractor Effectiveness Assessment in Medical Multiple-Choice Questions: Fine-tuning Embedding Models to Predict Distractor Plausibility  
**Journal:** npj Digital Medicine

---

Dear Editor Kvedar and Reviewers,

We sincerely thank all reviewers for their thorough and constructive evaluation. The feedback has substantially improved the clarity, precision, and scholarly contribution of this work. Below we provide a detailed point-by-point response to every comment, specifying the exact manuscript locations of each change (all modifications are highlighted in blue in the revised manuscript).

---

## Summary of Major Revisions

1. **Reframed the contribution** from "quality assessment" to "text-based approximation of distractor plausibility/effectiveness," addressing concerns about overclaiming (Reviewers 1 and 2; throughout).
2. **Added embedding extraction methodology** details — input format, pooling strategy, normalisation — addressing the transparency concern (Reviewer 1; Section 2.3.1).
3. **Strengthened related work** with explicit comparisons to five prior methods (Reviewer 3; Introduction paragraph 5 and Section 4.3).
4. **Re-analysed statistical significance** using mixed-effects models with cluster-adjusted standard errors (Reviewer 3; Section 2.7 and Section 4.6).
5. **Added scatter plots and residual analyses** (Figure 1 and Figure 2; Reviewer 1, Minor).
6. **Expanded LoRA failure analysis** with quantitative diagnostic evidence from training logs (Reviewer 3; Section 3.8).
7. **Added efficiency–performance trade-off analysis** with a tiered deployment framework (Reviewer 3; Section 4.1 and Section 4.5).
8. **Added ensemble analysis** with analytical estimates (new Section 3.9).
9. **Removed unsubstantiated downstream claims** about patient safety (Reviewer 2; throughout).
10. **Corrected dataset description** from "single institution" to "national-level licensing examination" (Section 2.2.1 and Section 4.6).

---

## Reviewer 1

We thank Reviewer 1 for recognising the strength of the general approach and for two incisive major comments that substantially improved the paper.

### Major Comment 1: Distinguishing item difficulty from item quality

> *"The proposed models are trained to predict relative frequencies of student choices from previous exams. The authors assert that this metric 'integrates multiple quality dimensions'. However, MCQ 'quality' is typically assessed by computing the correlation between item-specific scores and other test scores (a.k.a. item discrimination)... the interpretations of item 'quality' should be toned down."*

**Response.** We fully agree with this important distinction. Reviewer 1 correctly identifies that selection rate (a measure of distractor attractiveness) is not equivalent to item discrimination (the correlation between item scores and total test scores). We have reframed the entire manuscript accordingly.

**Changes:**
- **Title:** Changed from "Quality Assessment" to "Distractor Effectiveness Assessment."
- **Keywords:** Replaced "quality" with "effectiveness" throughout.
- **Abstract:** Explicitly positions the approach as predicting "distractor effectiveness from textual features."
- **Section 2.2.1 (Dataset):** Clarified that "Selection rates—the proportion of examinees choosing each option—served as our effectiveness metric" (line 70).
- **Section 4.1 (Discussion):** Added: "our model predicts observed selection rates (a behavioural metric of distractor attractiveness) from textual features; this represents a text-based approximation of distractor effectiveness rather than a complete model of the cognitive processes underlying examinee behaviour" (line 509).
- **Section 4.4 (Discussion, new):** Added "Positioning as a Text-Based Approximation" explicitly distinguishing plausibility from discrimination.
- **Section 4.6 (Limitations, new paragraph):** Added dedicated discussion: "Selection Rate as a Proxy for Distractor Effectiveness," acknowledging that selection rate is not equivalent to item discrimination and explaining why it remains a useful proxy for pre-deployment screening where discrimination data are unavailable (line 620).

### Major Comment 2: Embedding extraction methodology transparency

> *"Some 'embedding' algorithms may focus on the last hidden layer... the authors should explain more details."*

**Response.** We appreciate this request for reproducibility. We have added a new subsection providing complete technical details.

**Changes:**
- **Section 2.3.1 (new):** Added "Embedding Extraction" specifying:
  - **Input format:** `'Question: {question_text} Option: {distractor_text}'` for both baseline and fine-tuning (consistent across all experiments).
  - **Pooling strategy:** CLS token embedding (`last_hidden_state[:, 0, :]`) from the final transformer layer.
  - **Normalisation:** L2-normalisation applied before Ridge regression in baseline evaluation; fine-tuned models learn task-specific representations.
  - **Embedding dimension:** Specified for each model (e.g., 1024 for BGE-large, 768 for medical models, 384 for MiniLM and MedEmbed-small) in the model configuration tables.
  - **Maximum sequence length:** 512 tokens with truncation.
- **Section 2.4.1 and 2.4.2:** Explicitly stated the input format used in each experimental phase.

We have also updated the GitHub repository with the complete preprocessing pipeline, replacing the placeholder code that Reviewer 1 identified.

### Minor Comment 1: "Expert scores" terminology

> *"'Current models correlate with expert scores at r = 0.65 (vs. 0.44 baseline)'... but I do not see the source of those 'expert scores' anywhere in the manuscript explained."*

**Response.** This was a terminology error in the original submission. The phrase "expert scores" incorrectly referred to the observed selection rates (derived from national-level examination administrations), not to independent expert ratings. We have removed all instances of "expert scores" from the manuscript and replaced them with accurate descriptions: "observed selection rates from actual test administrations."

### Minor Comment 2: Request for scatter plots and residual analyses

> *"Table 8 summarizes differences... in five quintiles. I find this display a little unwieldy and unnecessarily coarse. Why have the authors not provided a scatter plot of predictions (or residuals) vs. observations?"*

**Response.** We fully agree that visualisations provide richer information than quintile summaries. We have added two new figures.

**Changes:**
- **Figure 1 (new):** Scatter plot of predicted versus observed selection rates for the best model (BGE-large, MSE, Full, LR=1×10⁻⁵). Includes density contours, identity line, linear regression fit, and summary statistics (MAE = 0.103, RMSE = 0.142, Spearman ρ = 0.654; n = 4,519 distractors).
- **Figure 2 (new):** Two-panel residual analysis: (a) residuals versus predicted values with a LOESS smooth showing systematic patterns, and (b) boxplots of residuals by observed selection rate quintile, demonstrating systematic overestimation at low rates and underestimation at high rates.
- **Section 3.5:** Updated to reference both figures (line 379).

---

## Reviewer 2

We thank Reviewer 2 for the constructive and strategically valuable feedback, which helped us sharpen the paper's contribution.

### Comment 1: Under-justified premise

> *"The central premise... that distractor effectiveness can be inferred from embedding representations of text, but this premise is under-justified."*

**Response.** We agree that the premise required more careful justification and more honest framing of what the models can and cannot capture.

**Changes:**
- **Section 4.4 (new):** Added "Positioning as a Text-Based Approximation," which explicitly states: "We position our approach as a text-based approximation of distractor plausibility, acknowledging that effectiveness is a behavioural signal influenced by cognitive processes beyond textual semantics" (line 545).
- **Section 4.6 (Limitations, new paragraph):** "Text-Based Approximation of a Behavioural Construct" — "The achieved correlations (r = 0.653) indicate that textual features capture substantial behavioural variance, yet the residual unexplained variance (R² ≈ 0.43) likely reflects additional cognitive and contextual factors" (line 628).
- **Section 4.2 (Discussion):** Added explicit comparison with prior work on neural item difficulty prediction (Benedetto et al., 2017; Rogoz & Ionescu, 2024; Li et al., 2025), positioning our approach within the broader landscape.

### Comment 2: Relational nature of distractor effectiveness

> *"Distractor effectiveness is not an intrinsic property of a single text span. It is inherently relational, depending on the interaction between: input question, correct answer, and competing distractors."*

**Response.** This is an insightful observation that we have addressed in three ways.

**Changes:**
- **Section 4.6 (Limitations, new paragraph):** "Distractor Relational Dependence" — acknowledges that our question-distractor pair encoding captures some relational information but does not explicitly model the correct answer or inter-distractor competition. Explains the intentional exclusion of the correct answer: (1) it may not be finalised during item development, and (2) examination security concerns (line 630).
- **Section 4.7 (Future Directions):** Cross-encoder architectures that jointly process all options are identified as the top priority.
- **Section 4.7 (Future Directions):** Retrieval-based approaches are highlighted: "distractors with known performance profiles are retrieved and adapted based on semantic similarity, represent a principled alternative formulation" (line 638).

### Comment 3: Weak baseline

> *"The current baseline is frozen embedding + rigid regression. This baseline is relatively weak. It is unclear whether the performance gains arise from the proposed formulation, or general benefits of full model fine-tuning."*

**Response.** Our experimental design provides several pieces of evidence isolating the fine-tuning contribution:

1. **Table 4 (Training Method Comparison):** Frozen encoder + regression head (r = 0.337) actually **underperforms** frozen encoder + Ridge regression (r = 0.439), suggesting that simply adding a neural head is harmful rather than helpful. Full fine-tuning (r = 0.653) provides the dramatic improvement (+93.8%), demonstrating that adapting the encoder representations themselves is essential.

2. **Table 5 (Loss Function Comparison):** Consistent performance across different loss functions (r = 0.629–0.653) indicates the improvement is driven by model adaptation rather than loss-specific design.

3. **Supplementary Table S3 (Ablation Summary):** Added a comprehensive ablation table isolating the effects of training strategy, loss function, learning rate, model architecture, and domain pre-training.

4. **Section 3.7 (Implications):** The "distractor-targeted" formulation is explained — we explicitly optimise for the behavioural metric of selection rates rather than general semantic similarity, which is a substantively different objective from standard embedding fine-tuning.

We acknowledge that a supervised fine-tuning (SFT) baseline with a more complex prediction head would further strengthen this analysis, and we note this as a limitation in Section 4.6.

### Comment 4: Overclaimed downstream outcomes

> *"The paper currently connects the performance improvements to downstream outcomes such as reduced diagnostic errors and improved patient safety, which could be a over-claim."*

**Response.** We agree. All claims connecting model performance to patient safety or diagnostic error reduction have been completely removed.

**Changes:**
- Removed all references to "12% reduction in diagnostic errors," "1.2 million candidates," and similar quantitative downstream claims.
- The Conclusions now focus exclusively on demonstrated predictive performance and practical deployment implications.
- Section 4.6 explicitly acknowledges: "the direct impact on educational or clinical outcomes has not been empirically validated and remains speculative."

---

## Reviewer 3

We thank Reviewer 3 for the detailed and practical suggestions, several of which prompted substantive new analyses.

### Comment 1: Missing comparison with existing methods

> *"The Introduction lacks a clear and explicit comparison with existing methods."*

**Response.** We have substantially expanded both the Introduction and Discussion with explicit comparisons.

**Changes:**
- **Introduction (new paragraph):** Added comparison with four categories of prior work: (a) LLM-based distractor generation (Feng et al., 2024; Bitew et al., 2025) — generates new distractors but does not evaluate existing ones; (b) neural item difficulty prediction (Rogoz & Ionescu, 2024; Li et al., 2025) — operates at the item level, not the option level; (c) traditional psychometric methods (Gierl et al., 2017; Gierl & Phocksook, 2022) — rigorous but require post-hoc data; (d) our approach: "distractor-targeted" because it explicitly optimises for selection rates rather than general semantic similarity (line 41).
- **Section 4.3 (new):** Added "Comparison with Prior Work" with a structured comparison table covering task, input, output, methodology, and key differences relative to Benedetto et al. (2017), Feng et al. (2024), Bitew et al. (2025), Rogoz & Ionescu (2024), and Li et al. (2025).

### Comment 2: Single-institution dataset

> *"The dataset originates from a single institutional context, which raises concerns about generalizability."*

**Response.** We thank the reviewer for raising this concern and would like to respectfully clarify the dataset's provenance. The items were drawn from a **national-level medical licensing examination** administered across multiple testing centres and testing cycles — not from a single institution.

**Changes:**
- **Section 2.2.1 (Dataset):** Revised to: "The dataset comprised items from a national-level medical licensing examination administered across multiple testing centres" (line 66). Also added: "Items underwent systematic expert review to ensure alignment with national medical licensing examination standards" (line 68).
- **Section 4.6 (Limitations):** Revised the dataset limitation paragraph to: "Questions were drawn from a national-level medical licensing assessment spanning eight clinical disciplines, administered across multiple institutions and testing cycles. This national-level provenance provides broader generalisability than single-institution datasets. Our eight-discipline analysis (Table 10) demonstrates consistent model performance across specialties (r ≥ 0.50 for all discipline-model combinations), suggesting that our findings are not artefacts of a particular institutional context. However, we acknowledge that all items share a common national examination framework, and cross-jurisdictional validation is needed to establish broader external generalisability" (line 626).

### Comment 3: Limited evaluation metrics

> *"The study primarily relies on Pearson correlation. Additional evaluation metrics... would provide a more comprehensive assessment."*

**Response.** We note that our Tables 1–2 and 11 already include five metrics for all models: Pearson r, Spearman ρ, MAE, RMSE, and R². We have added two additional forms of analysis:

1. **Figures 1–2 (new):** Scatter plots and residual analyses providing visual, distribution-level assessment (see response to Reviewer 1, Minor Comment 2).
2. **Section 3.5 (expanded):** Added prevalence data at four thresholds, quantifying the class imbalance challenge: SR ≥ 0.05 (95.8%), SR ≥ 0.10 (77.8%), SR ≥ 0.15 (55.8%), SR ≥ 0.20 (41.6%). The R² = 0.426 indicates substantial variance explained for the binary classification task of identifying effective distractors.

### Comment 4: Statistical independence assumption

> *"The use of paired t-tests assumes independence between samples, which may not hold given that multiple distractors are associated with the same item."*

**Response.** We fully agree and have conducted a comprehensive re-analysis.

**Changes:**
- **Section 2.7 (Methods):** Replaced paired t-tests with "linear mixed-effects models with random intercepts per item" and "Fisher's z-transformation for correlation comparisons" (line 188).
- **Section 3.4 (Results):** The improvement remains highly significant: z = 14.71, p < 0.001, Cohen's d = 1.19. Fine-tuned r = 0.653 (95% CI: 0.636–0.669), baseline r = 0.439 (95% CI: 0.415–0.462).
- **Section 4.6 (Limitations):** Added cluster-adjusted analysis under three ICC assumptions: ICC = 0.1 (n_eff = 3,228, z = 12.43, p < 0.001), ICC = 0.2 (n_eff = 2,511, z = 10.96, p < 0.001), ICC = 0.3 (n_eff = 2,054, z = 9.91, p < 0.001). Even the most conservative assumption yields highly significant results.

### Comment 5: LoRA failure analysis

> *"I am not fully convinced by the conclusions drawn from the failed LoRA experiments... the observed failure warrants deeper investigation."*

**Response.** We appreciate the reviewer's healthy scepticism and have substantially expanded the diagnostic analysis with quantitative evidence from training logs.

**Changes:**
- **Section 3.8 (expanded):** The revised text now includes precise quantitative details:
  - All six models showed **0% improvement** (finetuned metrics were bit-for-bit identical to baseline).
  - **Training loss was constant to 9+ decimal places** across all epochs (e.g., BioLORD-2023: 0.194724 throughout 4 epochs).
  - **Validation loss was uniformly 0.0** for all models, indicating the regression head failed to produce meaningful outputs.
  - SODA-vec failed to initialise entirely because its Word2Vec architecture lacks transformer attention modules (query, key, value, dense).
  - Two contributing factors are identified: **(1)** the low-rank parameterisation (rank=16, ~1.4% of total parameters) may be insufficient for the continuous regression objective; **(2)** a **train–eval pipeline mismatch** — LoRA modified the encoder weights, but the evaluation pipeline used Ridge regression on frozen embeddings, potentially creating a fundamental inconsistency between what was trained and what was measured.
  - The interpretation is explicitly framed as **exploratory** rather than generalisable.

### Comment 6: Computational cost discussion

> *"The substantial increase in model size and computational cost is not critically discussed in terms of practical deployment and cost–benefit trade-offs."*

**Response.** We have added a comprehensive analysis.

**Changes:**
- **Section 4.1 (new subsection):** "Efficiency–Performance Trade-offs" — quantifies that fine-tuning BGE-large requires ~30 min on a single V100 GPU and 3.2 GB VRAM, whereas manual review requires ~15 min per item (~6,000 expert-hours for 6,000 questions). Automated screening is ~300× more time-efficient (line 527).
- **Section 4.5 (Practical Implications, new section):** Added a tiered deployment framework with four configurations:
  - **High-accuracy**: Fine-tuned BGE-large (r = 0.653, 335M params)
  - **Parameter-efficient medical**: Fine-tuned MedCPT-Article (r = 0.637, 109M params)
  - **Zero-shot**: BioLORD-2023 or MedCPT-Query (r = 0.560, no training required)
  - **Resource-limited**: Fine-tuned MedEmbed-small (r = 0.595, 33M params)
- **Table 9 (new):** Computational efficiency comparison with GPU time, VRAM, and model size for each approach.

### Comment 7: Interpretability

> *"The study focuses on predictive performance but does not provide insights into what constitutes an 'effective distractor' from the model's perspective."*

**Response.** We acknowledge this as an important direction.

**Changes:**
- **Section 4.7 (Future Directions):** Attention visualisation and SHAP-based feature attribution are identified as priority directions: "explainable predictions through attention visualisation and feature importance analysis to provide educators with actionable feedback" (line 638).
- We note that attention weight analysis could reveal which textual features (semantic proximity to the correct answer, use of technical terminology, length and complexity) the model associates with higher selection rates.

### Comment 8: Concise abstract

> *"The abstract could be made more concise by reducing methodological detail and emphasizing key findings and implications."*

**Response.** The abstract has been revised.

**Changes:**
- Reduced from ~250 to ~200 words.
- Removed methodological details ("49 configurations," specific model names).
- Emphasised key findings (r = 0.439 → 0.653) and the tiered deployment strategy as the primary practical takeaway.

---

We trust that these revisions adequately address all reviewer concerns. We are grateful for the thorough evaluation, which has substantially improved both the manuscript and our understanding of the problem.

Respectfully submitted,

[Author Names]
