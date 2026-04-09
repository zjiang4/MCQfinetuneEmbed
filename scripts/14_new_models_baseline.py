#!/usr/bin/env python3
"""
Baseline Evaluation for New Embedding Models
Tests all models in newEmbeddingModels directory with frozen embeddings + Ridge regression
"""
import json
import logging
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(PROJECT_DIR / 'logs' / '14_new_models_baseline.log'))
    ],
    force=True
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data' / 'processed'
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'results' / 'new_models_baseline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_DIR / 'newEmbeddingModels'

LOG_DIR = PROJECT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'


def load_data(split: str) -> List[Dict]:
    path = DATA_DIR / f'{split}.json'
    with open(path) as f:
        data = json.load(f)
    return data.get('samples', data)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'Pearson_r': float(pr),
        'Pearson_p': float(pp),
        'Spearman_r': float(sr),
        'Spearman_p': float(sp),
    }


def create_training_data(samples: List[Dict]) -> Tuple[List[str], List[float]]:
    texts, rates = [], []
    for s in samples:
        q = s['question']
        opts = s['options']
        for opt in opts:
            if not opt['is_correct']:
                texts.append(f"Question: {q} Option: {opt['text']}")
                rates.append(opt['selection_rate'])
    return texts, rates


def get_model_info(model_path: str) -> Dict:
    config_path = Path(model_path) / 'config.json'
    info = {'path': model_path, 'hidden_size': None, 'vocab_size': None}
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                info['hidden_size'] = config.get('hidden_size', config.get('dim', None))
                info['vocab_size'] = config.get('vocab_size', None)
                info['max_position_embeddings'] = config.get('max_position_embeddings', None)
                info['model_type'] = config.get('model_type', 'unknown')
        except:
            pass
    
    return info


def extract_embeddings(model, tokenizer, texts: List[str], batch_size: int = 32, device: str = 'cpu') -> np.ndarray:
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings", leave=False):
            batch_texts = texts[i:i+batch_size]
            
            try:
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                outputs = model(**encoded)
                
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_emb = outputs.pooler_output.cpu().numpy()
                else:
                    batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_emb)
            except Exception as e:
                logger.warning(f"Error processing batch {i}: {e}")
                if embeddings:
                    emb_dim = embeddings[0].shape[1]
                    embeddings.append(np.zeros((len(batch_texts), emb_dim)))
                continue
    
    if not embeddings:
        return None
    
    embeddings = np.vstack(embeddings)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    
    return embeddings


def evaluate_baseline(model_path: str, test_data: List[Dict], model_name: str) -> Dict:
    logger.info(f"\nEvaluating: {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model_info = get_model_info(model_path)
        logger.info(f"Model info: {model_info}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        
        hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768
        logger.info(f"Hidden size: {hidden_size}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return {
            'model': model_name,
            'status': 'failed',
            'error': str(e)
        }
    
    test_texts, test_labels = create_training_data(test_data)
    logger.info(f"Test samples: {len(test_texts)}")
    
    logger.info("Extracting embeddings...")
    embeddings = extract_embeddings(model, tokenizer, test_texts, batch_size=32, device=device)
    
    if embeddings is None or len(embeddings) == 0:
        logger.error("Failed to extract embeddings")
        return {
            'model': model_name,
            'status': 'failed',
            'error': 'Failed to extract embeddings'
        }
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    labels = np.array(test_labels)
    
    logger.info("Running 5-fold cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kfold.split(embeddings)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        
        metrics = compute_metrics(y_test, y_pred)
        fold_results.append(metrics)
        
        logger.info(f"  Fold {fold+1}: Pearson={metrics['Pearson_r']:.4f}, MAE={metrics['MAE']:.4f}")
    
    mean_pearson = np.mean([r['Pearson_r'] for r in fold_results])
    std_pearson = np.std([r['Pearson_r'] for r in fold_results])
    mean_mae = np.mean([r['MAE'] for r in fold_results])
    mean_r2 = np.mean([r['R2'] for r in fold_results])
    
    logger.info(f"\nResults for {model_name}:")
    logger.info(f"  Pearson: {mean_pearson:.4f} (±{std_pearson:.4f})")
    logger.info(f"  MAE: {mean_mae:.4f}")
    logger.info(f"  R2: {mean_r2:.4f}")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'model': model_name,
        'model_info': model_info,
        'status': 'success',
        'Pearson_mean': float(mean_pearson),
        'Pearson_std': float(std_pearson),
        'MAE_mean': float(mean_mae),
        'R2_mean': float(mean_r2),
        'fold_results': fold_results,
        'n_test_samples': len(test_texts),
        'embedding_dim': int(embeddings.shape[1]),
    }


def main():
    logger.info("="*80)
    logger.info("BASELINE EVALUATION FOR NEW EMBEDDING MODELS")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now()}")
    
    test_data = load_data('test')
    logger.info(f"Test data loaded: {len(test_data)} questions")
    
    model_dirs = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir()])
    logger.info(f"\nFound {len(model_dirs)} models to evaluate:")
    for d in model_dirs:
        logger.info(f"  - {d.name}")
    
    all_results = []
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        model_path = str(model_dir)
        
        result = evaluate_baseline(model_path, test_data, model_name)
        all_results.append(result)
        
        result_file = OUTPUT_DIR / f"{model_name}_baseline.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved results to {result_file}")
    
    all_results_file = OUTPUT_DIR / 'all_baseline_results.json'
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    successful = [r for r in all_results if r.get('status') == 'success']
    failed = [r for r in all_results if r.get('status') == 'failed']
    
    logger.info(f"\nTotal: {len(all_results)}, Success: {len(successful)}, Failed: {len(failed)}")
    
    if failed:
        logger.info("\nFailed models:")
        for r in failed:
            logger.info(f"  - {r['model']}: {r.get('error', 'Unknown error')}")
    
    if successful:
        sorted_results = sorted(successful, key=lambda x: x['Pearson_mean'], reverse=True)
        
        logger.info("\nResults ranked by Pearson correlation:")
        logger.info("-" * 70)
        logger.info(f"{'Rank':<6}{'Model':<45}{'Pearson':<12}{'MAE':<10}")
        logger.info("-" * 70)
        
        for i, r in enumerate(sorted_results, 1):
            logger.info(f"{i:<6}{r['model']:<45}{r['Pearson_mean']:.4f}±{r['Pearson_std']:.3f}  {r['MAE_mean']:.4f}")
    
    logger.info(f"\nCompleted: {datetime.now()}")
    logger.info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
