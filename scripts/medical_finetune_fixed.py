#!/usr/bin/env python3
"""
Medical Embedding Fine-tuning - FIXED VERSION
Based on successful E5 configuration (Pearson 0.73)

Key fixes:
1. Use MSELoss (proven to work)
2. Full fine-tuning (no LoRA)
3. Proper loss computation
4. Correct validation logic

Models:
- BioLORD-2023
- MedCPT-Query-Encoder
- MedCPT-Article-Encoder
- MedEmbed-small-v0.1
- SapBERT-PubMed
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

# Force single GPU to avoid DataParallel StopIteration bug
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(Path(__file__).resolve().parent.parent / 'logs' / 'medical_finetune_fixed.log'))
    ]
)
logger = logging.getLogger(__name__)

# Medical models configuration
MEDICAL_MODELS = [
    {
        'name': 'FremyCompany/BioLORD-2023',
        'short_name': 'biolord-2023',
        'hidden_dim': 768,
        'lr': 2e-5,
        'batch_size': 16,
        'prefix': ''
    },
    {
        'name': 'ncbi/MedCPT-Query-Encoder',
        'short_name': 'medcpt-query',
        'hidden_dim': 768,
        'lr': 2e-5,
        'batch_size': 16,
        'prefix': ''
    },
    {
        'name': 'ncbi/MedCPT-Article-Encoder',
        'short_name': 'medcpt-article',
        'hidden_dim': 768,
        'lr': 2e-5,
        'batch_size': 16,
        'prefix': ''
    },
    {
        'name': 'abhinand/MedEmbed-small-v0.1',
        'short_name': 'medembed-small',
        'hidden_dim': 384,
        'lr': 3e-5,
        'batch_size': 32,
        'prefix': ''
    },
    {
        'name': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        'short_name': 'sapbert-pubmed',
        'hidden_dim': 768,
        'lr': 2e-5,
        'batch_size': 16,
        'prefix': ''
    },
]


def load_data() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load train/val/test data."""
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    with open(data_dir / 'train.json') as f:
        train_data = json.load(f)
    with open(data_dir / 'val.json') as f:
        val_data = json.load(f)
    with open(data_dir / 'test.json') as f:
        test_data = json.load(f)
    
    train = train_data.get('samples', train_data)
    val = val_data.get('samples', val_data)
    test = test_data.get('samples', test_data)
    
    logger.info(f"Loaded: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute evaluation metrics."""
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


class MedicalEmbeddingFineTuner:
    """Fine-tuner for medical embedding models."""
    
    def __init__(self, model_config: Dict):
        self.config = model_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model using sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        
        model_name = self.config['name']
        logger.info(f"Loading {model_name}...")
        
        # Try local path first
        local_path = Path(__file__).parent.parent / 'newEmbeddingModels' / model_name.replace('/', '__')
        
        # Use model_kwargs to ensure proper dtype and avoid StopIteration bug
        model_kwargs = {'torch_dtype': torch.float32}
        
        if local_path.exists():
            logger.info(f"Loading from local: {local_path}")
            self.model = SentenceTransformer(str(local_path), device='cpu', model_kwargs=model_kwargs)
        else:
            logger.info(f"Loading from HuggingFace: {model_name}")
            self.model = SentenceTransformer(model_name, device='cpu', model_kwargs=model_kwargs)
        
        # Move to single GPU after loading to avoid DataParallel
        self.model = self.model.to('cuda:0')
        
        logger.info(f"Model {self.config['short_name']} loaded successfully on cuda:0")
    
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts to embeddings."""
        prefix = self.config.get('prefix', '')
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def extract_features(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for Ridge regression."""
        X_list, y_list = [], []
        
        for s in tqdm(samples, desc="Extracting features"):
            q = s['question']
            opts = s['options']
            
            # Find correct answer
            correct = next((o for o in opts if o['is_correct']), None)
            if not correct:
                continue
            
            # Encode question and correct answer
            q_emb = self.encode([q])[0]
            c_emb = self.encode([correct['text']])[0]
            
            # Process each distractor
            for opt in opts:
                if not opt['is_correct']:
                    d_emb = self.encode([opt['text']])[0]
                    
                    # Compute similarities
                    sim_qd = np.dot(q_emb, d_emb)
                    sim_dc = np.dot(d_emb, c_emb)
                    sim_qc = np.dot(q_emb, c_emb)
                    
                    # Feature vector
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
        """Evaluate baseline (zero-shot) performance."""
        logger.info(f"Evaluating baseline for {self.config['short_name']}...")
        
        X, y = self.extract_features(samples)
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        metrics_list = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_test)
            
            # Compute metrics
            m = compute_metrics(y_test, y_pred)
            metrics_list.append(m)
            logger.info(f"  Fold {fold+1}: Pearson={m['Pearson_r']:.4f}, MAE={m['MAE']:.4f}")
        
        # Aggregate results
        results = {
            'Pearson_mean': float(np.mean([m['Pearson_r'] for m in metrics_list])),
            'Pearson_std': float(np.std([m['Pearson_r'] for m in metrics_list])),
            'MAE_mean': float(np.mean([m['MAE'] for m in metrics_list])),
            'RMSE_mean': float(np.mean([m['RMSE'] for m in metrics_list])),
            'R2_mean': float(np.mean([m['R2'] for m in metrics_list])),
            'Spearman_mean': float(np.mean([m['Spearman_r'] for m in metrics_list])),
            'n_folds': n_folds,
            'n_samples': len(y)
        }
        
        logger.info(
            f"Baseline: Pearson={results['Pearson_mean']:.4f}±{results['Pearson_std']:.4f}, "
            f"MAE={results['MAE_mean']:.4f}"
        )
        
        return results
    
    def finetune(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict:
        """Fine-tune model using CosineSimilarityLoss."""
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader
        
        logger.info(f"Fine-tuning {self.config['short_name']} with CosineSimilarityLoss...")
        
        # Prepare training examples
        # For CosineSimilarityLoss, we need pairs of sentences with similarity scores
        train_examples = []
        for s in train_samples:
            q = s['question']
            for opt in s['options']:
                if not opt['is_correct']:
                    # Create sentence pair: question and distractor
                    # Label is the selection rate (0-1), which we'll use as similarity
                    train_examples.append(
                        InputExample(texts=[q, opt['text']], label=float(opt['selection_rate']))
                    )
        
        logger.info(f"Training examples: {len(train_examples)}")
        
        # Create data loader
        train_loader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.config['batch_size']
        )
        
        # CosineSimilarityLoss for sentence pairs
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Training parameters
        epochs = 10
        patience = 3
        best_pearson = -1.0
        patience_counter = 0
        history = {'train_loss': [], 'val_pearson': []}
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train one epoch with multi_gpu=False to avoid DataParallel issues
            self.model.fit(
                train_objectives=[(train_loader, train_loss)],
                epochs=1,
                warmup_steps=len(train_loader) // 10,
                show_progress_bar=True,
                optimizer_params={'lr': self.config['lr']},
                use_amp=False,  # Disable automatic mixed precision
                checkpoint_path=None,
                checkpoint_save_steps=0,
                checkpoint_save_total_limit=0
            )
            
            # Validate
            val_results = self.evaluate_baseline(val_samples, n_folds=3)
            val_pearson = val_results['Pearson_mean']
            history['val_pearson'].append(val_pearson)
            
            logger.info(f"Epoch {epoch+1}: val_pearson={val_pearson:.4f}")
            
            # Early stopping
            if val_pearson > best_pearson:
                best_pearson = val_pearson
                patience_counter = 0
                # Save best model
                self._save_checkpoint('best')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        output_dir = (
            Path(__file__).parent.parent / 'models' / 'medical_finetuned_fixed' /
            f"{self.config['short_name']}_{name}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))
        logger.info(f"Saved checkpoint to {output_dir}")


def run_experiment(model_config: Dict, train: List[Dict], val: List[Dict], test: List[Dict]) -> Dict:
    """Run complete experiment for one model."""
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT: {model_config['name']}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Initialize fine-tuner
        tuner = MedicalEmbeddingFineTuner(model_config)
        
        # Evaluate baseline
        logger.info("Step 1: Baseline evaluation")
        baseline_results = tuner.evaluate_baseline(test, n_folds=5)
        
        # Fine-tune
        logger.info("Step 2: Fine-tuning")
        history = tuner.finetune(train, val)
        
        # Evaluate fine-tuned model
        logger.info("Step 3: Fine-tuned evaluation")
        finetuned_results = tuner.evaluate_baseline(test, n_folds=5)
        
        # Compute improvement
        improvement = {
            'pearson_delta': finetuned_results['Pearson_mean'] - baseline_results['Pearson_mean'],
            'pearson_percent': (
                (finetuned_results['Pearson_mean'] - baseline_results['Pearson_mean']) /
                baseline_results['Pearson_mean'] * 100
            )
        }
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            'model_name': model_config['name'],
            'short_name': model_config['short_name'],
            'baseline': baseline_results,
            'finetuned': finetuned_results,
            'improvement': improvement,
            'history': history,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✓ SUCCESS: {model_config['short_name']}")
        logger.info(f"  Baseline: {baseline_results['Pearson_mean']:.4f}")
        logger.info(f"  Fine-tuned: {finetuned_results['Pearson_mean']:.4f}")
        logger.info(f"  Improvement: +{improvement['pearson_delta']:.4f} ({improvement['pearson_percent']:.1f}%)")
        logger.info(f"  Duration: {duration/60:.1f} minutes")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ FAILED: {model_config['short_name']}")
        logger.error(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'model_name': model_config['name'],
            'short_name': model_config['short_name'],
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("MEDICAL EMBEDDING FINE-TUNING - FIXED VERSION")
    logger.info("=" * 80)
    
    # Load data
    train, val, test = load_data()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'outputs' / 'results' / 'medical_fixed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments for all models
    all_results = []
    
    for model_config in MEDICAL_MODELS:
        result = run_experiment(model_config, train, val, test)
        all_results.append(result)
        
        output_file = output_dir / f"{model_config['short_name']}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved result to {output_file}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_models': len(MEDICAL_MODELS),
        'results': all_results
    }
    
    summary_file = output_dir / 'all_results_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Summary saved to {summary_file}")
    logger.info("=" * 80)
    
    # Print summary table
    logger.info("\nRESULTS SUMMARY:")
    logger.info("-" * 80)
    logger.info(f"{'Model':<25} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<15}")
    logger.info("-" * 80)
    
    for r in all_results:
        if 'error' not in r:
            baseline = r['baseline']['Pearson_mean']
            finetuned = r['finetuned']['Pearson_mean']
            improvement = r['improvement']['pearson_percent']
            logger.info(
                f"{r['short_name']:<25} {baseline:<12.4f} {finetuned:<12.4f} "
                f"+{improvement:<14.1f}%"
            )
        else:
            logger.info(f"{r['short_name']:<25} FAILED: {r['error']}")
    
    logger.info("-" * 80)


if __name__ == '__main__':
    main()
