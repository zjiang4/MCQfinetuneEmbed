# Data Dictionary

## Dataset: Medical MCQ Distractor Quality Assessment

---

## Overview

This dataset contains 6,000 medical multiple-choice questions (MCQs) with observed distractor selection rates, designed for automated quality assessment using natural language processing.


## Samples

```json
   {
      "id": "Rheumatology_518",
      "question": "A 58-year-old man presents with sudden onset of severe pain, swelling, and redness of the right first metatarsophalangeal joint that began overnight. He has a history of hypertension treated with hydrochlorothiazide and drinks alcohol regularly. Joint aspiration reveals needle-shaped, negatively birefringent crystals under polarized light microscopy. What is the most appropriate first-line treatment for this acute attack?",
      "content_area": "Rheumatology",
      "options": [
        {
          "text": "Allopurinol initiation during the acute attack",
          "is_correct": false,
          "selection_rate": 0.28,
          "has_valid_text": true
        },
        {
          "text": "Intra-articular hyaluronic acid injection",
          "is_correct": false,
          "selection_rate": 0.2,
          "has_valid_text": true
        },
        {
          "text": "Nonsteroidal anti-inflammatory drugs (NSAIDs)",
          "is_correct": true,
          "selection_rate": 0.3,
          "has_valid_text": true
        },
        {
          "text": "Methotrexate therapy",
          "is_correct": false,
          "selection_rate": 0.12,
          "has_valid_text": true
        },
        {
          "text": "Long-term colchicine prophylaxis only",
          "is_correct": false,
          "selection_rate": 0.1,
          "has_valid_text": true
        }
      ],
      "explanation": "This patient has classic acute gout, confirmed by negatively birefringent monosodium urate crystals. First-line therapy for an acute gout flare includes NSAIDs, colchicine, or corticosteroids. Urate-lowering therapy such as allopurinol should not be initiated during an acute attack, as it may worsen symptoms, but can be started after the flare has resolved.",
      "key_learning_points": "Rheumatology; Acute gout presents with sudden monoarticular arthritis, commonly affecting the first metatarsophalangeal joint, and is treated initially with NSAIDs, colchicine, or corticosteroids.",
      "domain": "Rheumatology"
    }
```

---

## Data Schema

### Question Object

```json
{
  "id": "string",
  "question": "string",
  "content_area": "string",
  "domain": "string",
  "options": [
    {
      "text": "string",
      "is_correct": "boolean",
      "selection_rate": "float",
      "has_valid_text": "boolean"
    }
  ]
}
```

---

## Field Descriptions

### Top-Level Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique identifier | "Cardiology_123" |
| `question` | string | Full question text | "A 65-year-old patient presents with..." |
| `content_area` | string | Clinical specialty | "Cardiology" |
| `domain` | string | Knowledge domain | "Cardiovascular" |
| `options` | array | List of answer options | See below |

### Option Fields

| Field | Type | Description | Range/Values |
|-------|------|-------------|--------------|
| `text` | string | Option text | Any valid text |
| `is_correct` | boolean | Correct answer flag | true/false |
| `selection_rate` | float | Proportion of examinees selecting this option | 0.0 - 1.0 |
| `has_valid_text` | boolean | Text validity flag | true/false |

---

## Clinical Disciplines

The dataset covers 8 clinical disciplines:

| Discipline | Code | Description | Sample Count |
|------------|------|-------------|--------------|
| Cardiology | CARD | Cardiovascular system | 750 questions |
| Endocrinology | ENDO | Endocrine system | 750 questions |
| Haematology | HAEM | Blood and blood-forming tissues | 750 questions |
| Infectious Diseases | ID | Infectious disease | 750 questions |
| Nephrology | NEPH | Kidney and urinary system | 750 questions |
| Neurology | NEUR | Nervous system | 750 questions |
| Respiratory | RESP | Respiratory system | 750 questions |
| Rheumatology | RHEU | Musculoskeletal system | 750 questions |

---

## Data Splits

### Training Set
- **File**: `train.json`
- **Size**: 4,200 questions (70%)
- **Purpose**: Model training and parameter optimization

### Validation Set
- **File**: `val.json`
- **Size**: 896 questions (15%)
- **Purpose**: Hyperparameter tuning and early stopping

### Test Set
- **File**: `test.json`
- **Size**: 904 questions (15%)
- **Purpose**: Final model evaluation

---

## Statistics

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Questions | 6,000 |
| Total Options | 28,332 |
| Total Distractors | ~16,800 |
| Questions per Discipline | 750 |
| Options per Question | 3-5 |
| Distractors per Question | 2.8 (mean) |

### Selection Rate Distribution

| Statistic | Value |
|-----------|-------|
| Minimum | 0.00 |
| Maximum | 0.36 |
| Mean | 0.136 |
| Median | 0.125 |
| Std Dev | 0.068 |
| Skewness | 0.82 (right-skewed) |

### Selection Rate by Quintile

| Quintile | Range | Count | Percentage |
|----------|-------|-------|------------|
| Q1 (Very Low) | 0.00 - 0.07 | 723 | 20% |
| Q2 (Low) | 0.07 - 0.11 | 723 | 20% |
| Q3 (Moderate) | 0.11 - 0.15 | 723 | 20% |
| Q4 (High) | 0.15 - 0.21 | 723 | 20% |
| Q5 (Very High) | 0.21 - 0.36 | 723 | 20% |

---

## Quality Indicators

### Distractor Quality Criteria

**High-Quality Distractor**:
- Selection rate: 0.10 - 0.20
- Attracts partially knowledgeable students
- Discriminates between knowledge levels

**Low-Quality Distractor**:
- Selection rate: < 0.05 (too easy) or > 0.30 (too attractive)
- Random guessing or misleading

---

## Data Quality

### Validation Checks

1. **Completeness**
   - All questions have valid text
   - All options have valid text
   - Selection rates present for all options

2. **Consistency**
   - Exactly one correct answer per question
   - Selection rates sum to 1.0 per question
   - No duplicate options

3. **Validity**
   - Medical terminology verified
   - Clinical scenarios realistic
   - Expert-reviewed content

---

## Usage Guidelines

### For Model Training

**Input Features**:
- Question text
- Distractor text
- Correct answer text

**Target Variable**:
- Distractor selection rate (continuous, 0-1)

**Preprocessing**:
```python
# Example feature construction
features = {
    'question': sample['question'],
    'distractor': option['text'],
    'correct_answer': correct_option['text'],
    'target': option['selection_rate']
}
```

---

## Ethical Considerations

### Data Privacy
- No patient information included
- Questions are de-identified
- Institutional approval obtained

### Usage Restrictions
- For research purposes only
- Not for commercial use
- Proper citation required

---

## Citation

```bibtex
@dataset{mcq_distractor_2026,
  title={Medical MCQ Distractor Quality Assessment Dataset},
  author={[Author Names]},
  year={2026},
  publisher={[Institution]},
  version={1.0}
}
```

---

## Contact

For data access requests or questions:
- Email: [corresponding author email]
- Institution: [Institution Name]

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | March 2026 | Initial release |

---

**Last Updated**: March 2, 2026
