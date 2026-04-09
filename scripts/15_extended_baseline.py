#!/usr/bin/env python3
"""
Extended Baseline Evaluation - Include all working models
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
        logging.FileHandler(str(PROJECT_DIR / 'logs' / '15_extended_baseline.log'))
    ],
    force=True
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data' / 'processed'
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'results' / 'extended_baseline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_DIR / 'newEmbeddingModels'

LOG_DIR = PROJECT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

ALL_MODELS = {
    'Original Models': {
        'BAAI/bge-large-en-v1.5': str(Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--BAAI--bge-large-en-v1.5' / 'snapshots' / 'd4aa6901d3a41ba39fb536a557fa166f842b0e09'),
        'sentence-transformers/all-mpnet-base-v2': str(Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--sentence-transformers--all-mpnet-base-v2' / 'snapshots' / '9a3d5d55a1a4b94f1b6b80f08d8f60a3e2b20e1'),
    },
    'Medical Models (Previously Tested)': {
        'FremyCompany/BioLORD-2023': str(MODEL_DIR / 'FremyCompany__BioLORD-2023'),
        'abhinand/MedEmbed-small-v0.1': str(MODEL_DIR / 'abhinand__MedEmbed-small-v0.1'),
        'ncbi/MedCPT-Query-Encoder': str(MODEL_DIR / 'ncbi__MedCPT-Query-Encoder'),
        'cambridgeltl/SapBERT-from-PubMedBERT-fulltext': str(MODEL_DIR / 'cambridgeltl__SapBERT-from-PubMedBERT-fulltext'),
        'EMBO/soda-vec-negative-sampling': str(MODEL_DIR / 'EMBO__soda-vec-negative-sampling'),
        'ncbi/MedCPT-Article-Encoder': str(MODEL_DIR / 'ncbi__MedCPT-Article-Encoder'),
    },
    'Medical Models (Newly Working)': {
        'emilyalsentzer/Bio_ClinicalBERT': str(MODEL_DIR / 'emilyalsentzer__Bio_ClinicalBERT'),
        'sentence-transformers/embeddinggemma-300m-medical': str(MODEL_DIR / 'sentence-transformers__embeddinggemma-300m-medical'),
    }
}

def load_data(split: str) -> List[Dict]:
    path = DATA_DIR / f'{split}.json'
    with open(path) as f:
        data = json.load(f)
    return data.get('samples', data)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'Pearson_r': float(pearsonr(y_true, y_pred)[0]),
        'Spearman_r': float(spearmanr(y_true, y_pred)[0]),
    }

def create_training_data(samples: List[Dict]) -> Tuple[List[str], List[float]]:
    texts, rates = [], []
    for s in samples:
        for opt in s['options']:
            if not opt['is_correct']:
                texts.append(f"Question: {s['question']} Option: {opt['text']}")
                rates.append(opt['selection_rate'])
    return texts, rates

def evaluate_model(model_path: str, test_data: List[Dict], model_name: str) -> Dict:
    logger.info(f"\nEvaluating: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        hidden_size = model.config.hidden_size
        logger.info(f"  Hidden size: {hidden_size}")
    except Exception as e:
        logger.error(f"  Failed to load: {e}")
        return {'model': model_name, 'status': 'failed', 'error': str(e)}
    
    test_texts, test_labels = create_training_data(test_data)
    logger.info(f"  Test samples: {len(test_texts)}")
    
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_texts), batch_size), desc="  Encoding", leave=False):
            batch = test_texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                embeddings.append(out.pooler_output.cpu().numpy())
            else:
                embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    labels = np.array(test_labels)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(embeddings)):
        ridge = Ridge(alpha=1.0).fit(embeddings[train_idx], labels[train_idx])
        pred = ridge.predict(embeddings[test_idx])
        metrics = compute_metrics(labels[test_idx], pred)
        fold_results.append(metrics)
        logger.info(f"    Fold {fold+1}: Pearson={metrics['Pearson_r']:.4f}")
    
    result = {
        'model': model_name,
        'status': 'success',
        'Pearson_mean': float(np.mean([r['Pearson_r'] for r in fold_results])),
        'Pearson_std': float(np.std([r['Pearson_r'] for r in fold_results])),
        'MAE_mean': float(np.mean([r['MAE'] for r in fold_results])),
        'R2_mean': float(np.mean([r['R2'] for r in fold_results])),
    }
    
    logger.info(f"  Result: Pearson={result['Pearson_mean']:.4f}±{result['Pearson_std']:.3f}")
    
    del model
    torch.cuda.empty_cache()
    
    return result

def main():
    logger.info("="*80)
    logger.info("EXTENDED BASELINE EVALUATION")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now()}")
    
    test_data = load_data('test')
    logger.info(f"Test data: {len(test_data)} questions")
    
    all_results = []
    
    for category, models in ALL_MODELS.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"CATEGORY: {category}")
        logger.info(f"{'='*80}")
        
        for model_name, model_path in models.items():
            result = evaluate_model(model_path, test_data, model_name)
            result['category'] = category
            all_results.append(result)
            
            result_file = OUTPUT_DIR / f"{model_name.replace('/', '__')}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
    
    with open(OUTPUT_DIR / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    successful = [r for r in all_results if r.get('status') == 'success']
    failed = [r for r in all_results if r.get('status') == 'failed']
    
    logger.info(f"\nTotal: {len(all_results)}, Success: {len(successful)}, Failed: {len(failed)}")
    
    if successful:
        sorted_results = sorted(successful, key=lambda x: x['Pearson_mean'], reverse=True)
        logger.info("\nRanked by Pearson correlation:")
        logger.info("-"*80)
        for i, r in enumerate(sorted_results, 1):
            logger.info(f"{i:2d}. {r['model']:<50} {r['Pearson_mean']:.4f}±{r['Pearson_std']:.3f}")
    
    if failed:
        logger.info("\nFailed models:")
        for r in failed:
            logger.info(f"  - {r['model']}")
    
    logger.info(f"\nCompleted: {datetime.now()}")

if __name__ == '__main__':
    main()
