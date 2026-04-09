# Revision Plan: Data Consistency Audit & Fix

**Date**: April 9, 2026  
**Status**: Pending User Approval  
**Submission ID**: b6f32430-e205-4ad2-8932-f899b055b2e3

---

## Part A: Audit Summary — What's Wrong

### CRITICAL 1: Table 2 FT column is systematically scrambled

Table 2 title says "MSE Loss, Full Fine-tuning, LR = 2×10⁻⁵" but the FT values don't match LR=2e-5 from any source:

| Model | Current FT | Actual LR=2e-5 | Actual Origin of Current Value |
|-------|-----------|---------------|------------------------------|
| BGE-large | **0.649** | **0.637** | This is BGE-base's optimal value (0.649) |
| BGE-base | **0.638** | **0.649** | This is BGE-base LR=1e-5 (0.638) |
| E5-large | 0.610 | **0.608** | Off by 0.002; actually E5 optimal (LR=1e-5) |
| MPNet | **0.641** | **0.621** | This is BGE-large Huber LR=1e-5 (0.641) |
| MiniLM | **0.602** | **0.612** | No exact match; possibly transcription error |

R² column is also wrong: BGE-large R²=0.419 is actually BGE-base's R².

**Impact**: Abstract "22.0% to 32.7%", Section 3.2 text, Table 12 comparison.

### CRITICAL 2: Table 11 mixes two different experiment pipelines

- **Baseline column**: From docx Table 1 (Ridge regression, CLS token pooling, `baseline_all` experiment)
- **Fine-tuned column**: From `medical_fixed` experiment (CosineSimilarityLoss, sentence-pair input)

These use different evaluation methods, so the improvement % is an apples-to-oranges comparison.

| Model | Table 1 baseline | medical_fixed baseline | Gap |
|-------|-----------------|----------------------|-----|
| BioLORD | 0.560 | 0.485 | +0.075 |
| MedCPT-Query | 0.560 | 0.490 | +0.070 |
| MedCPT-Article | 0.536 | 0.445 | +0.091 |
| MedEmbed | 0.534 | 0.485 | +0.049 |
| SapBERT | 0.504 | 0.417 | +0.087 |

### CRITICAL 3: Test set size contradiction

- Data partitioning table (line 80): **3,615** distractor samples
- Figure 1 description (line 379): **n = 4,519** distractors
- Source data `comprehensive_v2`: n_samples = **3,615**
- `data/processed/test.json` after filtering: **4,519** valid distractor records

### CRITICAL 4: False claim "r ≥ 0.50 for all discipline-model combinations"

Table 10 shows BGE-base at Nephrology = 0.489 and Rheumatology = 0.492, both < 0.50.

### MODERATE 5: R² = r² throughout

docx consistently reports R² as r² (e.g., 0.653² = 0.426 ≈ 0.425), not regression R² (0.375). This is a convention choice, not an error, but should be noted.

### MODERATE 6: Cohen's d inconsistency

- Manuscript: d = 1.19
- Supplementary Materials: d = 1.08

### MODERATE 7: Conclusions precision

"r = 0.63-0.65" vs Table 5 "r = 0.629-0.653"

### MODERATE 8: README and docx still have removed claims

- README says "r ≥ 0.60 across 8 clinical domains" (removed from manuscript)
- README says "48.6% improvement" (correct, but needs context of which comparison)
- docx still has patient safety claims (removed from manuscript)

---

## Part B: Proposed Fixes

### Fix 1: Table 2 — Replace FT column with correct LR=2e-5 values

**Source**: docx Table 5 (Learning Rate Sensitivity) and `comprehensive_v2` for BGE-large/MPNet

| Model | Baseline r | FT r (LR=2e-5) | Gain | Improvement | MAE | RMSE |
|-------|-----------|----------------|------|-------------|-----|------|
| BGE-large | 0.514 | 0.637 | +0.123 | +23.9% | 0.041 | 0.054 |
| BGE-base | 0.504 | 0.649 | +0.145 | +28.8% | 0.040 | 0.052 |
| E5-large | 0.500 | 0.608 | +0.108 | +21.6% | 0.041 | 0.053 |
| MPNet | 0.483 | 0.621 | +0.138 | +28.6% | 0.041 | 0.053 |
| MiniLM | 0.458 | 0.612 | +0.154 | +33.6% | 0.042 | 0.054 |

**Note added**: "Baseline r values in this table reflect the Phase 2 evaluation pipeline (single-feature CLS embeddings) and may differ slightly from Table 1 (multi-feature baseline including correct answer embeddings) due to different train/test splits."

**Cascading changes**:
- Abstract: "22.0% to 32.7%" → "21.6% to 33.6%"
- Section 3.2: Update discussion text
- Table 12: BGE-large row — baseline stays 0.439 (from comprehensive_v2 pipeline), FT stays 0.653, improvement stays 48.6% (NO CHANGE to Table 12)
- Conclusions point 1: "r: 0.44 → 0.65, +48.6%" (NO CHANGE — this references Table 12)

### Fix 2: Table 11 — Use medical_fixed internal baselines

**Replace** the baseline column with `medical_fixed` experiment's own baselines:

| Model | Baseline | Fine-tuned | Gain | Improvement |
|-------|----------|-----------|------|-------------|
| MedCPT-Article | 0.445 | 0.637 | +0.192 | +43.2% |
| BioLORD-2023 | 0.485 | 0.617 | +0.132 | +27.3% |
| MedCPT-Query | 0.490 | 0.614 | +0.123 | +25.1% |
| SapBERT | 0.417 | 0.547 | +0.130 | +31.1% |
| MedEmbed-small | 0.485 | 0.595 | +0.110 | +22.7% |

**Note added**: "Baseline values were computed within the same experiment pipeline (CosineSimilarityLoss fine-tuning framework) to ensure comparable evaluation conditions."

**Cascading changes**:
- Abstract: "MedCPT-Article r = 0.637 with 67% fewer parameters" — NO CHANGE (FT value unchanged)
- Abstract: "+11.1% vs. best general baseline" — NO CHANGE (this refers to Table 1 zero-shot comparison)
- Table 12 medical row: Baseline changes from 0.536 to 0.445, Improvement from +18.8% to +43.2%
- Section 3.8 text: Update improvement ranges "8.5% to 18.8%" → "22.7% to 43.2%"

### Fix 3: Figure 1 n = 4,519 → n = 3,615

Single find-and-replace.

### Fix 4: "r ≥ 0.50" → "r ≥ 0.49"

Change to: "consistent model performance across specialties (r ≥ 0.49 for all discipline-model combinations, with all medical model combinations achieving r ≥ 0.54)"

### Fix 5: R² — Keep as r², add footnote

Add note: "R² reported as squared Pearson correlation (r²) rather than regression R²."

### Fix 6: Cohen's d — Unify to 1.19

Update Supplementary Materials d=1.08 → d=1.19 throughout.

### Fix 7: Conclusions precision

"r = 0.63-0.65" → "r = 0.629-0.653"

### Fix 8: README — Update to match revised manuscript

Remove outdated claims, update statistics, add link to paper.

---

## Part C: Anti-Reviewer-Attack Checklist

After fixes, verify these won't invite new criticism:

| Check | Status | Notes |
|-------|--------|-------|
| All tables internally consistent | After Fix 1-2 | Each table's baseline and FT from same pipeline |
| Cross-table comparisons explicitly noted | After Fix 1 | Footnote explains different pipelines |
| No absolute claims without qualification | After Fix 4 | "r ≥ 0.49" with actual range |
| Statistical method matches data structure | Already done | Mixed-effects model with cluster adjustment |
| No patient safety/downstream claims | Already done | Removed in prior revision |
| Dataset description accurate | Already done | National-level, not single institution |
| Model list in README matches paper | After Fix 8 | 5 general + 5 medical |
| GitHub code matches methodology | Needs update | Push real scripts |
| All cited numbers exist in tables | After all fixes | Full cross-reference check |
| Improvement % mathematically correct | After Fix 1-2 | Recalculate and verify |

---

## Part D: Files to Modify

| File | Changes |
|------|---------|
| `paper/npj_digital_medicine_v3_with_medical_embeddings.md` | Fixes 1-7 |
| `paper/SUPPLEMENTARY_MATERIALS.md` | Fix 6 (Cohen's d), Fix 1 cascade |
| `paper/RESPONSE_TO_REVIEWERS.md` | Update line numbers |
| `Publication Package/appendix/Supplementary_Materials.md` | Same as SUPPLEMENTARY_MATERIALS.md |
| GitHub `README.md` | Fix 8 |
| GitHub scripts | Push real experimental scripts |

---

## Part E: What Does NOT Change

These are already correct and should remain untouched:

- **Table 1** (baseline values from docx Table 0 / comprehensive_v2) — CORRECT
- **Table 3** (48 configurations from comprehensive_v2) — CORRECT
- **Table 4** (training method comparison) — CORRECT
- **Table 5** (loss function comparison, BGE-large LR=1e-5) — CORRECT
- **Table 6** (learning rate sensitivity) — CORRECT
- **Table 7** (model architecture comparison) — CORRECT
- **Table 8** (error analysis quintiles) — CORRECT
- **Table 9** (computational efficiency) — CORRECT
- **Table 10** (discipline-specific performance) — CORRECT
- **Table 12** (general vs medical comparison) — BGE-large row CORRECT, medical row needs Fix 2 cascade
- **Table 13** (ensemble estimates) — CORRECT
- **48.6% improvement** claim — CORRECT (0.439 → 0.653 from same pipeline)
- **Abstract key numbers** — Mostly CORRECT, only "22.0% to 32.7%" needs update
- **Statistical significance** (z=14.71, d=1.19) — CORRECT

---

**Awaiting your approval to execute.**
