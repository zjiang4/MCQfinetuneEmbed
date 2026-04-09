#!/usr/bin/env python3
"""
Discipline-Specific Baseline Analysis

Evaluates top baseline models on each of the 8 clinical disciplines to determine
if medical embeddings have differential advantages across specialties.

Outputs:
- outputs/results/discipline_analysis/discipline_performance.json
- outputs/results/discipline_analysis/discipline_summary.csv
"""

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
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

OUTPUT_DIR = Path(__file__).parent.parent / 'outputs' / 'results' / 'discipline_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_BASELINE_MODELS = [
    {
        'name': 'FremyCompany/BioLORD-2023',
        'short_name': 'BioLORD-2023',
        'category': 'Medical',
        'prefix': '',
        'dim': 768
    },
    {
        'name': 'ncbi/MedCPT-Query-Encoder',
        'short_name': 'MedCPT-Query',
        'category': 'Medical',
        'prefix': '',
        'dim': 768
    },
    {
        'name': 'BAAI/bge-base-en-v1.5',
        'short_name': 'BGE-base',
        'category': 'General',
        'prefix': '',
        'dim': 768
    }
]

DISCIPLINES = [
    'Cardiology',
    'Endocrinology',
    'Haematology',
    'Infectious Diseases',
    'Nephrology',
    'Neurology',
    'Respiratory',
    'Rheumatology'
]


def load_data(split: str = 'test') -> List[Dict]:
    """Load processed data."""
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / f'{split}.json'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('samples', data)


def prepare_discipline_pairs(samples: List[Dict], discipline: str) -> List[Dict]:
    """Prepare evaluation pairs for a specific discipline."""
    pairs = []
    
    for sample in samples:
        if sample.get('content_area') != discipline:
            continue
            
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


def compute_embeddings(pairs: List[Dict], model, prefix: str = '') -> tuple:
    """Compute embeddings for question-distractor-correct triplets."""
    texts_q = [prefix + p['question'] for p in pairs]
    texts_d = [prefix + p['distractor'] for p in pairs]
    texts_c = [prefix + p['correct'] for p in pairs]
    
    emb_q = model.encode(texts_q, show_progress_bar=False, normalize_embeddings=True)
    emb_d = model.encode(texts_d, show_progress_bar=False, normalize_embeddings=True)
    emb_c = model.encode(texts_c, show_progress_bar=False, normalize_embeddings=True)
    
    features = np.concatenate([
        emb_q,
        emb_d,
        emb_c,
        emb_q * emb_d,
        emb_d * emb_c,
        np.abs(emb_q - emb_d),
        np.abs(emb_d - emb_c)
    ], axis=1)
    
    targets = np.array([p['selection_rate'] for p in pairs])
    
    return features, targets


def evaluate_discipline(
    model_config: Dict,
    pairs: List[Dict],
    n_folds: int = 5
) -> Dict:
    """Evaluate model on a specific discipline using k-fold CV."""
    from sentence_transformers import SentenceTransformer
    
    model_name = model_config['name']
    short_name = model_config['short_name']
    prefix = model_config.get('prefix', '')
    
    logger.info(f"  Evaluating {short_name}...")
    
    model = SentenceTransformer(model_name, device='cuda')
    
    features, targets = compute_embeddings(pairs, model, prefix)
    
    del model
    import torch
    torch.cuda.empty_cache()
    
    fold_results = []
    n_samples = len(targets)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n_samples // n_folds
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_train, X_val = features[train_indices], features[val_indices]
        y_train, y_val = targets[train_indices], targets[val_indices]
        
        regressor = Ridge(alpha=1.0)
        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_val)
        
        predictions = np.clip(predictions, 0, 1)
        
        pearson_r, _ = pearsonr(predictions, y_val)
        spearman_r, _ = spearmanr(predictions, y_val)
        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)
        
        fold_results.append({
            'fold': fold + 1,
            'pearson': float(pearson_r),
            'spearman': float(spearman_r),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'n_samples': int(len(y_val))
        })
    
    pearson_scores = [r['pearson'] for r in fold_results]
    mae_scores = [r['mae'] for r in fold_results]
    r2_scores = [r['r2'] for r in fold_results]
    
    return {
        'model': short_name,
        'category': model_config['category'],
        'pearson_mean': float(np.mean(pearson_scores)),
        'pearson_std': float(np.std(pearson_scores)),
        'mae_mean': float(np.mean(mae_scores)),
        'mae_std': float(np.std(mae_scores)),
        'r2_mean': float(np.mean(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
        'fold_results': fold_results,
        'n_samples': int(n_samples)
    }


def main():
    """Run discipline-specific analysis."""
    logger.info("=" * 80)
    logger.info("DISCIPLINE-SPECIFIC BASELINE ANALYSIS")
    logger.info("=" * 80)
    
    test_data = load_data('test')
    logger.info(f"Loaded {len(test_data)} test samples")
    
    all_results = {}
    
    for discipline in DISCIPLINES:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"DISCIPLINE: {discipline}")
        logger.info(f"{'=' * 80}")
        
        discipline_pairs = prepare_discipline_pairs(test_data, discipline)
        logger.info(f"Found {len(discipline_pairs)} distractor samples")
        
        if len(discipline_pairs) < 50:
            logger.warning(f"Skipping {discipline} - insufficient samples")
            continue
        
        discipline_results = {
            'discipline': discipline,
            'n_samples': len(discipline_pairs),
            'models': {}
        }
        
        for model_config in TOP_BASELINE_MODELS:
            try:
                result = evaluate_discipline(model_config, discipline_pairs)
                discipline_results['models'][model_config['short_name']] = result
                
                logger.info(f"    {model_config['short_name']}: r={result['pearson_mean']:.4f} (±{result['pearson_std']:.4f})")
                
            except Exception as e:
                logger.error(f"    Failed to evaluate {model_config['short_name']}: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[discipline] = discipline_results
    
    logger.info(f"\n{'=' * 80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'=' * 80}")
    
    output_json = OUTPUT_DIR / 'discipline_performance.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved detailed results to {output_json}")
    
    summary_data = []
    for discipline, disc_results in all_results.items():
        for model_name, model_results in disc_results['models'].items():
            summary_data.append({
                'Discipline': discipline,
                'Model': model_name,
                'Category': model_results['category'],
                'Pearson r': model_results['pearson_mean'],
                'Pearson Std': model_results['pearson_std'],
                'MAE': model_results['mae_mean'],
                'R²': model_results['r2_mean'],
                'N Samples': model_results['n_samples']
            })
    
    df_summary = pd.DataFrame(summary_data)
    output_csv = OUTPUT_DIR / 'discipline_summary.csv'
    df_summary.to_csv(output_csv, index=False)
    logger.info(f"Saved summary CSV to {output_csv}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("DISCIPLINE COMPARISON SUMMARY")
    logger.info(f"{'=' * 80}")
    
    pivot_table = df_summary.pivot_table(
        values='Pearson r',
        index='Discipline',
        columns='Model',
        aggfunc='first'
    )
    logger.info("\n" + pivot_table.to_string())
    
    medical_advantage = []
    for discipline in DISCIPLINES:
        if discipline in all_results:
            models = all_results[discipline]['models']
            if 'BioLORD-2023' in models and 'BGE-base' in models:
                med_r = models['BioLORD-2023']['pearson_mean']
                gen_r = models['BGE-base']['pearson_mean']
                advantage = ((med_r - gen_r) / gen_r) * 100
                medical_advantage.append({
                    'Discipline': discipline,
                    'Medical (BioLORD)': med_r,
                    'General (BGE-base)': gen_r,
                    'Advantage (%)': advantage
                })
    
    df_advantage = pd.DataFrame(medical_advantage)
    logger.info("\nMedical Embedding Advantage by Discipline:")
    logger.info(df_advantage.to_string(index=False))
    
    logger.info(f"\n✓ Analysis complete! Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
