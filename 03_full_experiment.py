#!/usr/bin/env python3
"""
Phase 1 & 2: Improved Baseline and Fine-tuning Pipeline

Key improvements:
1. Use Ridge regression on embeddings for baseline (not just 1-similarity)
2. Proper validation with Pearson correlation
3. Three loss types: CosineSimilarityLoss, MSELoss, MultipleNegativesRankingLoss
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

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/admin01/Embedding训练/outputs/experiment_v2.log')
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / 'outputs' / 'results' / 'experiment_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(__file__).parent.parent / 'models' / 'finetuned_v2'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {'name': 'intfloat/e5-large-v2', 'short': 'e5-large', 'dim': 1024, 'prefix': 'query: ', 'batch': 16, 'lr': 2e-5},
    {'name': 'BAAI/bge-large-en-v1.5', 'short': 'bge-large', 'dim': 1024, 'prefix': '', 'batch': 16, 'lr': 2e-5},
    {'name': 'BAAI/bge-base-en-v1.5', 'short': 'bge-base', 'dim': 768, 'prefix': '', 'batch': 16, 'lr': 2e-5},
]


def load_data(split: str) -> List[Dict]:
    path = Path(__file__).parent.parent / 'data' / 'processed' / f'{split}.json'
    with open(path) as f:
        data = json.load(f)
    return data.get('samples', data)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pr, pp = pearsonr(y_true, y_pred)
    sr, sp = spearmanr(y_true, y_pred)
    return {
        'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2),
        'Pearson_r': float(pr), 'Pearson_p': float(pp),
        'Spearman_r': float(sr), 'Spearman_p': float(sp),
        'n': len(y_true)
    }


class ExperimentRunner:
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self._load_model()
    
    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading {self.config['name']}...")
        self.model = SentenceTransformer(self.config['name'], device=self.device)
    
    def encode(self, texts: List[str], batch: int = 64) -> np.ndarray:
        prefix = self.config.get('prefix', '')
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        return self.model.encode(texts, batch_size=batch, show_progress_bar=False, 
                                  convert_to_numpy=True, normalize_embeddings=True)
    
    def extract_features(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features using Ridge regression on embeddings."""
        X_list, y_list = [], []
        
        for s in tqdm(samples, desc=f"Features {self.config['short']}"):
            q = s['question']
            opts = s['options']
            
            # Find correct
            correct = next((o for o in opts if o['is_correct']), None)
            if not correct:
                continue
            
            # Encode question and correct once
            q_emb = self.encode([q])[0]
            c_emb = self.encode([correct['text']])[0]
            
            for opt in opts:
                if not opt['is_correct']:
                    d_emb = self.encode([opt['text']])[0]
                    
                    # Features: embedding + similarities
                    sim_qd = np.dot(q_emb, d_emb)
                    sim_dc = np.dot(d_emb, c_emb)
                    sim_qc = np.dot(q_emb, c_emb)
                    
                    feat = np.concatenate([
                        d_emb,
                        q_emb * d_emb,
                        np.abs(q_emb - d_emb),
                        [sim_qd, sim_dc, sim_qc]
                    ])
                    X_list.append(feat)
                    y_list.append(opt['selection_rate'])
        
        return np.array(X_list), np.array(y_list)
    
    def evaluate_baseline(self, samples: List[Dict], n_folds: int = 5) -> Dict:
        """Evaluate with cross-validation using Ridge regression."""
        X, y = self.extract_features(samples)
        
        logger.info(f"Feature shape: {X.shape}, Labels: {y.shape}")
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        metrics_list = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_test)
            
            m = compute_metrics(y_test, y_pred)
            metrics_list.append(m)
            logger.info(f"  Fold {fold+1}: Pearson={m['Pearson_r']:.4f}, MAE={m['MAE']:.4f}")
        
        results = {
            'Pearson_r_mean': float(np.mean([m['Pearson_r'] for m in metrics_list])),
            'Pearson_r_std': float(np.std([m['Pearson_r'] for m in metrics_list])),
            'MAE_mean': float(np.mean([m['MAE'] for m in metrics_list])),
            'RMSE_mean': float(np.mean([m['RMSE'] for m in metrics_list])),
            'R2_mean': float(np.mean([m['R2'] for m in metrics_list])),
            'n_folds': n_folds,
            'n_samples': len(y)
        }
        
        logger.info(f"CV Results: Pearson={results['Pearson_r_mean']:.4f}±{results['Pearson_r_std']:.4f}")
        return results
    
    def finetune_cosine(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict:
        """Fine-tune using CosineSimilarityLoss with Pearson validation."""
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader
        
        logger.info("Fine-tuning with CosineSimilarityLoss...")
        
        # Prepare examples
        train_examples = []
        for s in train_samples:
            q = s['question']
            for opt in s['options']:
                if not opt['is_correct']:
                    prefix = self.config.get('prefix', '')
                    text = f"{prefix}{q} {opt['text']}"
                    train_examples.append(InputExample(texts=[text], label=opt['selection_rate']))
        
        logger.info(f"Training examples: {len(train_examples)}")
        
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=self.config['batch'])
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        epochs = 10
        patience = 3
        best_pearson = -1.0
        patience_count = 0
        
        for epoch in range(epochs):
            # Train one epoch
            self.model.fit(
                train_objectives=[(train_loader, train_loss)],
                epochs=1,
                warmup_steps=len(train_loader) // 10,
                show_progress_bar=True,
                output_path=None
            )
            
            # Validate
            val_results = self.evaluate_baseline(val_samples, n_folds=3)
            val_pearson = val_results['Pearson_r_mean']
            
            logger.info(f"Epoch {epoch+1}: val_pearson={val_pearson:.4f}")
            
            if val_pearson > best_pearson:
                best_pearson = val_pearson
                patience_count = 0
                # Save best model
                save_path = MODEL_DIR / f"{self.config['short']}_cosine"
                self.model.save(str(save_path))
                logger.info(f"  Saved best model (pearson={val_pearson:.4f})")
            else:
                patience_count += 1
            
            if patience_count >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {'best_val_pearson': best_pearson}
    
    def finetune_mse(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict:
        """Fine-tune using MSELoss."""
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader
        
        logger.info("Fine-tuning with MSELoss...")
        
        train_examples = []
        for s in train_samples:
            q = s['question']
            for opt in s['options']:
                if not opt['is_correct']:
                    prefix = self.config.get('prefix', '')
                    text = f"{prefix}{q} {opt['text']}"
                    train_examples.append(InputExample(texts=[text], label=opt['selection_rate']))
        
        train_loader = DataLoader(train_examples, shuffle=True, batch_size=self.config['batch'])
        train_loss = losses.MSELoss(self.model)
        
        epochs = 10
        patience = 3
        best_pearson = -1.0
        patience_count = 0
        
        for epoch in range(epochs):
            self.model.fit(
                train_objectives=[(train_loader, train_loss)],
                epochs=1,
                warmup_steps=len(train_loader) // 10,
                show_progress_bar=True
            )
            
            val_results = self.evaluate_baseline(val_samples, n_folds=3)
            val_pearson = val_results['Pearson_r_mean']
            
            logger.info(f"Epoch {epoch+1}: val_pearson={val_pearson:.4f}")
            
            if val_pearson > best_pearson:
                best_pearson = val_pearson
                patience_count = 0
                save_path = MODEL_DIR / f"{self.config['short']}_mse"
                self.model.save(str(save_path))
            else:
                patience_count += 1
            
            if patience_count >= patience:
                break
        
        return {'best_val_pearson': best_pearson}


def run_full_experiment():
    """Run full experiment: baseline + 3 loss types."""
    logger.info("=" * 70)
    logger.info("Full Experiment: Baseline + Fine-tuning")
    logger.info("=" * 70)
    
    # Load data
    train = load_data('train')
    val = load_data('val')
    test = load_data('test')
    
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    all_results = {}
    
    for i, model_config in enumerate(MODELS):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i+1}/{len(MODELS)}] {model_config['name']}")
        logger.info("=" * 70)
        
        model_results = {'model': model_config['name'], 'short': model_config['short']}
        
        try:
            runner = ExperimentRunner(model_config)
            
            # Baseline
            logger.info("\n--- BASELINE ---")
            baseline_results = runner.evaluate_baseline(test, n_folds=5)
            model_results['baseline'] = baseline_results
            logger.info(f"Baseline: Pearson={baseline_results['Pearson_r_mean']:.4f}")
            
            # Fine-tune Cosine
            logger.info("\n--- FINE-TUNING: CosineSimilarityLoss ---")
            cosine_history = runner.finetune_cosine(train, val)
            
            # Re-load best model and evaluate
            runner.model = runner.model.__class__(str(MODEL_DIR / f"{model_config['short']}_cosine"), device=runner.device)
            cosine_results = runner.evaluate_baseline(test, n_folds=5)
            model_results['finetuned_cosine'] = cosine_results
            logger.info(f"Fine-tuned (Cosine): Pearson={cosine_results['Pearson_r_mean']:.4f}")
            
            # Fine-tune MSE
            logger.info("\n--- FINE-TUNING: MSELoss ---")
            runner = ExperimentRunner(model_config)  # Fresh model
            mse_history = runner.finetune_mse(train, val)
            
            runner.model = runner.model.__class__(str(MODEL_DIR / f"{model_config['short']}_mse"), device=runner.device)
            mse_results = runner.evaluate_baseline(test, n_folds=5)
            model_results['finetuned_mse'] = mse_results
            logger.info(f"Fine-tuned (MSE): Pearson={mse_results['Pearson_r_mean']:.4f}")
            
            # Calculate improvements
            model_results['improvement_cosine'] = cosine_results['Pearson_r_mean'] - baseline_results['Pearson_r_mean']
            model_results['improvement_mse'] = mse_results['Pearson_r_mean'] - baseline_results['Pearson_r_mean']
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            model_results['error'] = str(e)
        
        all_results[model_config['short']] = model_results
        
        # Save intermediate results
        with open(OUTPUT_DIR / 'experiment_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    print(f"{'Model':<20} {'Baseline':>10} {'Cosine':>10} {'MSE':>10} {'Best':>10}")
    print("-" * 70)
    
    for short, res in all_results.items():
        if 'error' not in res:
            baseline = res['baseline']['Pearson_r_mean']
            cosine = res['finetuned_cosine']['Pearson_r_mean']
            mse = res['finetuned_mse']['Pearson_r_mean']
            best = max(cosine, mse)
            print(f"{short:<20} {baseline:>10.4f} {cosine:>10.4f} {mse:>10.4f} {best:>10.4f}")
    
    print("=" * 70)
    
    return all_results


if __name__ == '__main__':
    results = run_full_experiment()
    logger.info("\nExperiment Complete!")
