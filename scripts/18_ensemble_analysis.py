#!/usr/bin/env python3
"""
Ensemble Analysis: Combining Medical and General Embeddings

Tests whether medical and general embeddings capture complementary information
by evaluating ensemble approaches.

Outputs:
- outputs/results/ensemble_analysis/ensemble_performance.json
- outputs/results/ensemble_analysis/ensemble_summary.csv
"""

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / 'outputs' / 'results' / 'ensemble_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_MODELS = {
    'BioLORD-2023': {
        'name': 'FremyCompany/BioLORD-2023',
        'category': 'Medical',
        'type': 'baseline'
    },
    'MedCPT-Query': {
        'name': 'ncbi/MedCPT-Query-Encoder',
        'category': 'Medical',
        'type': 'baseline'
    },
    'BGE-base': {
        'name': 'BAAI/bge-base-en-v1.5',
        'category': 'General',
        'type': 'baseline'
    }
}

FINETUNED_MODEL = {
    'name': 'BAAI/bge-large-en-v1.5',
    'checkpoint_path': 'models/finetuned/bge-large-en-v1.5_best',
    'category': 'General',
    'type': 'finetuned'
}


def load_data(split: str = 'test') -> List[Dict]:
    """Load processed data."""
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / f'{split}.json'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('samples', data)


def prepare_evaluation_pairs(samples: List[Dict]) -> List[Dict]:
    """Prepare evaluation pairs."""
    pairs = []
    
    for sample in samples:
        question = sample['question']
        options = sample['options']
        
        correct_opt = next((o for o in options if o['is_correct']), None)
        if not correct_opt:
            continue
        
        correct_text = correct_opt['text']
        
        for opt in options:
            if not opt['is_correct']:
                pairs.append({
                    'question': question,
                    'distractor': opt['text'],
                    'correct': correct_text,
                    'selection_rate': opt['selection_rate'],
                    'content_area': sample.get('content_area', 'Unknown')
                })
    
    return pairs


def compute_model_predictions(
    pairs: List[Dict],
    model,
    prefix: str = '',
    use_finetuned: bool = False
) -> np.ndarray:
    """Compute predictions from a single model."""
    from sentence_transformers import SentenceTransformer
    import torch
    import torch.nn as nn
    
    if use_finetuned:
        texts_q = [p['question'] for p in pairs]
        texts_d = [p['distractor'] for p in pairs]
        texts_c = [p['correct'] for p in pairs]
        
        emb_q = model.encode(texts_q, show_progress_bar=False, normalize_embeddings=True)
        emb_d = model.encode(texts_d, show_progress_bar=False, normalize_embeddings=True)
        emb_c = model.encode(texts_c, show_progress_bar=False, normalize_embeddings=True)
        
        features = np.concatenate([
            emb_q, emb_d, emb_c,
            emb_q * emb_d, emb_d * emb_c,
            np.abs(emb_q - emb_d), np.abs(emb_d - emb_c)
        ], axis=1)
        
        regressor = Ridge(alpha=1.0)
        targets = np.array([p['selection_rate'] for p in pairs])
        regressor.fit(features, targets)
        predictions = regressor.predict(features)
        predictions = np.clip(predictions, 0, 1)
        
    else:
        texts_q = [prefix + p['question'] for p in pairs]
        texts_d = [prefix + p['distractor'] for p in pairs]
        texts_c = [prefix + p['correct'] for p in pairs]
        
        emb_q = model.encode(texts_q, show_progress_bar=False, normalize_embeddings=True)
        emb_d = model.encode(texts_d, show_progress_bar=False, normalize_embeddings=True)
        emb_c = model.encode(texts_c, show_progress_bar=False, normalize_embeddings=True)
        
        features = np.concatenate([
            emb_q, emb_d, emb_c,
            emb_q * emb_d, emb_d * emb_c,
            np.abs(emb_q - emb_d), np.abs(emb_d - emb_c)
        ], axis=1)
        
        targets = np.array([p['selection_rate'] for p in pairs])
        
        n_samples = len(targets)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        fold_size = n_samples // 5
        all_predictions = np.zeros(n_samples)
        
        for fold in range(5):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < 4 else n_samples
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train, X_val = features[train_indices], features[val_indices]
            y_train = targets[train_indices]
            
            regressor = Ridge(alpha=1.0)
            regressor.fit(X_train, y_train)
            fold_predictions = regressor.predict(X_val)
            all_predictions[val_indices] = fold_predictions
        
        predictions = all_predictions
    
    return predictions


def evaluate_ensemble(
    predictions_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    ensemble_name: str,
    models_included: List[str]
) -> Dict:
    """Evaluate an ensemble of models using simple averaging."""
    ensemble_pred = np.mean([predictions_dict[m] for m in models_included], axis=0)
    ensemble_pred = np.clip(ensemble_pred, 0, 1)
    
    pearson_r, _ = pearsonr(ensemble_pred, targets)
    spearman_r, _ = spearmanr(ensemble_pred, targets)
    mae = mean_absolute_error(targets, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(targets, ensemble_pred))
    r2 = r2_score(targets, ensemble_pred)
    
    return {
        'ensemble_name': ensemble_name,
        'models': models_included,
        'pearson': float(pearson_r),
        'spearman': float(spearman_r),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }


def main():
    """Run ensemble analysis."""
    logger.info("=" * 80)
    logger.info("ENSEMBLE ANALYSIS: MEDICAL + GENERAL EMBEDDINGS")
    logger.info("=" * 80)
    
    test_data = load_data('test')
    logger.info(f"Loaded {len(test_data)} test samples")
    
    pairs = prepare_evaluation_pairs(test_data)
    logger.info(f"Prepared {len(pairs)} evaluation pairs")
    
    targets = np.array([p['selection_rate'] for p in pairs])
    
    all_predictions = {}
    
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 1: Loading baseline models and computing predictions")
    logger.info(f"{'=' * 80}")
    
    from sentence_transformers import SentenceTransformer
    
    for short_name, config in BASELINE_MODELS.items():
        logger.info(f"\nLoading {short_name} ({config['category']})...")
        
        try:
            model = SentenceTransformer(config['name'], device='cuda')
            predictions = compute_model_predictions(pairs, model, prefix='', use_finetuned=False)
            all_predictions[short_name] = predictions
            
            del model
            import torch
            torch.cuda.empty_cache()
            
            pearson_r, _ = pearsonr(predictions, targets)
            logger.info(f"  Single model Pearson r: {pearson_r:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load {short_name}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 2: Evaluating ensemble combinations")
    logger.info(f"{'=' * 80}")
    
    ensemble_configs = [
        {
            'name': 'BioLORD + BGE-base (average)',
            'models': ['BioLORD-2023', 'BGE-base']
        },
        {
            'name': 'MedCPT + BGE-base (average)',
            'models': ['MedCPT-Query', 'BGE-base']
        },
        {
            'name': 'All Medical (average)',
            'models': ['BioLORD-2023', 'MedCPT-Query']
        }
    ]
    
    ensemble_results = []
    
    for config in ensemble_configs:
        if all(m in all_predictions for m in config['models']):
            logger.info(f"\nEvaluating: {config['name']}")
            result = evaluate_ensemble(
                all_predictions,
                targets,
                config['name'],
                config['models']
            )
            ensemble_results.append(result)
            
            logger.info(f"  Ensemble Pearson r: {result['pearson']:.4f}")
            
            best_single = max([all_predictions[m] for m in config['models']], 
                             key=lambda p: pearsonr(p, targets)[0])
            best_single_r, _ = pearsonr(best_single, targets)
            improvement = ((result['pearson'] - best_single_r) / best_single_r) * 100
            logger.info(f"  Improvement over best single: {improvement:+.2f}%")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("STEP 3: Fine-tuned + Medical ensemble")
    logger.info(f"{'=' * 80}")
    
    finetuned_path = Path(__file__).parent.parent / FINETUNED_MODEL['checkpoint_path']
    
    if finetuned_path.exists():
        logger.info(f"\nLoading fine-tuned BGE-large...")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            import torch.nn as nn
            
            tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL['name'])
            base_model = AutoModel.from_pretrained(FINETUNED_MODEL['name'])
            
            class FineTunedModel(nn.Module):
                def __init__(self, base_model, output_dim=1):
                    super().__init__()
                    self.base_model = base_model
                    hidden_size = base_model.config.hidden_size
                    
                    self.regressor = nn.Sequential(
                        nn.Linear(hidden_size * 7, 256),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(256, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, input_ids, attention_mask):
                    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    return self.regressor(embeddings)
            
            finetuned_model = FineTunedModel(base_model)
            finetuned_model.load_state_dict(torch.load(finetuned_path / 'pytorch_model.bin'))
            finetuned_model.to('cuda')
            finetuned_model.eval()
            
            logger.info("Computing fine-tuned predictions...")
            
            batch_size = 32
            finetuned_preds = []
            
            for i in tqdm(range(0, len(pairs), batch_size)):
                batch = pairs[i:i+batch_size]
                
                texts_q = [p['question'] for p in batch]
                texts_d = [p['distractor'] for p in batch]
                texts_c = [p['correct'] for p in batch]
                
                combined_texts = [f"{q} [SEP] {d} [SEP] {c}" for q, d, c in zip(texts_q, texts_d, texts_c)]
                
                inputs = tokenizer(
                    combined_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to('cuda')
                
                with torch.no_grad():
                    outputs = finetuned_model(**inputs)
                    finetuned_preds.extend(outputs.cpu().numpy().flatten())
            
            all_predictions['BGE-large-FT'] = np.array(finetuned_preds)
            
            pearson_r, _ = pearsonr(all_predictions['BGE-large-FT'], targets)
            logger.info(f"  Fine-tuned BGE-large Pearson r: {pearson_r:.4f}")
            
            logger.info("\nEvaluating fine-tuned + medical ensembles...")
            
            ft_ensemble_configs = [
                {
                    'name': 'BGE-large (FT) + BioLORD',
                    'models': ['BGE-large-FT', 'BioLORD-2023']
                },
                {
                    'name': 'BGE-large (FT) + MedCPT',
                    'models': ['BGE-large-FT', 'MedCPT-Query']
                }
            ]
            
            for config in ft_ensemble_configs:
                if all(m in all_predictions for m in config['models']):
                    result = evaluate_ensemble(
                        all_predictions,
                        targets,
                        config['name'],
                        config['models']
                    )
                    ensemble_results.append(result)
                    
                    logger.info(f"\n{config['name']}:")
                    logger.info(f"  Ensemble Pearson r: {result['pearson']:.4f}")
                    
                    ft_r, _ = pearsonr(all_predictions['BGE-large-FT'], targets)
                    improvement = ((result['pearson'] - ft_r) / ft_r) * 100
                    logger.info(f"  Improvement over fine-tuned alone: {improvement:+.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning(f"Fine-tuned model not found at {finetuned_path}")
        logger.info("Skipping fine-tuned + medical ensemble analysis")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'=' * 80}")
    
    output_json = OUTPUT_DIR / 'ensemble_performance.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(pairs),
            'ensemble_results': ensemble_results,
            'single_model_results': {
                name: {
                    'pearson': float(pearsonr(preds, targets)[0]),
                    'mae': float(mean_absolute_error(targets, preds))
                }
                for name, preds in all_predictions.items()
            }
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {output_json}")
    
    summary_data = []
    for result in ensemble_results:
        summary_data.append({
            'Ensemble': result['ensemble_name'],
            'Models': ', '.join(result['models']),
            'Pearson r': result['pearson'],
            'Spearman ρ': result['spearman'],
            'MAE': result['mae'],
            'RMSE': result['rmse'],
            'R²': result['r2']
        })
    
    df_summary = pd.DataFrame(summary_data)
    output_csv = OUTPUT_DIR / 'ensemble_summary.csv'
    df_summary.to_csv(output_csv, index=False)
    logger.info(f"Saved summary CSV to {output_csv}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("ENSEMBLE PERFORMANCE SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info("\n" + df_summary.to_string(index=False))
    
    logger.info(f"\n✓ Analysis complete! Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
