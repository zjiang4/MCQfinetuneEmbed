#!/usr/bin/env python3
"""
Fine-tune All Baseline Models with Optimized Contrastive Learning.

Models to fine-tune:
1. E5-large-v2 (best zero-shot: Pearson 0.1803)
2. BGE-base-en (second best: Pearson 0.1585)
3. BGE-large-en (Pearson 0.1470)

Optimizations:
- Better loss function (margin ranking + contrastive)
- Learning rate warmup + cosine decay
- Hard negative mining
- Early stopping with patience
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
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
        logging.FileHandler('/home/admin01/Embedding训练/outputs/finetune_all.log')
    ]
)
logger = logging.getLogger(__name__)

MODELS_TO_FINETUNE = [
    {
        'name': 'intfloat/e5-large-v2',
        'short_name': 'e5-large-v2',
        'dim': 1024,
        'lr': 2e-5,
        'batch_size': 8,
        'prefix': 'query: ',
        'description': 'Microsoft E5 Large v2 - Best zero-shot'
    },
    {
        'name': 'BAAI/bge-base-en-v1.5',
        'short_name': 'bge-base-en',
        'dim': 768,
        'lr': 2e-5,
        'batch_size': 16,
        'prefix': '',
        'description': 'BGE Base English v1.5 - Fast and effective'
    },
    {
        'name': 'BAAI/bge-large-en-v1.5',
        'short_name': 'bge-large-en',
        'dim': 1024,
        'lr': 1e-5,
        'batch_size': 8,
        'prefix': '',
        'description': 'BGE Large English v1.5'
    },
]


def load_all_data() -> List[Dict]:
    """Load full dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'full_dataset.json'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', data)
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def prepare_pairs(samples: List[Dict]) -> List[Dict]:
    """Prepare training pairs."""
    pairs = []
    for sample in samples:
        question = sample['question']
        options = sample['options']
        
        correct = None
        distractors = []
        distractor_rates = []
        
        for opt in options:
            if opt.get('is_correct', False):
                correct = opt['text']
            else:
                distractors.append(opt['text'])
                distractor_rates.append(opt.get('selection_rate', 0.0))
        
        if correct and len(distractors) >= 2:
            pairs.append({
                'question': question,
                'correct': correct,
                'distractors': distractors,
                'selection_rates': distractor_rates,
                'id': sample.get('id', '')
            })
    
    logger.info(f"Prepared {len(pairs)} training pairs")
    return pairs


class ImprovedContrastiveLoss(nn.Module):
    """Improved contrastive loss with margin ranking."""
    
    def __init__(self, margin: float = 0.3, temperature: float = 0.05):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        q_emb: torch.Tensor,
        d_embs: torch.Tensor,
        c_emb: torch.Tensor,
        rates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute improved contrastive loss.
        
        Args:
            q_emb: Question embedding [D]
            d_embs: Distractor embeddings [N, D]
            c_emb: Correct answer embedding [D]
            rates: Selection rates [N]
        """
        n = d_embs.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=q_emb.device, requires_grad=True)
        
        d_q_sims = F.cosine_similarity(d_embs.unsqueeze(0), q_emb.unsqueeze(0).unsqueeze(0))
        d_c_sims = F.cosine_similarity(d_embs.unsqueeze(0), c_emb.unsqueeze(0).unsqueeze(0))
        
        anchor = 0.5
        quality = torch.abs(rates - anchor)
        
        loss = torch.tensor(0.0, device=q_emb.device, requires_grad=True)
        n_pairs = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    high_i = quality[i] > quality[j]
                    if high_i:
                        sim_diff = d_q_sims[j] - d_q_sims[i]
                        loss = loss + F.relu(self.margin + sim_diff)
                        n_pairs += 1
        
        if n_pairs > 0:
            loss = loss / n_pairs
        
        return loss


class QualityPredictorLoss(nn.Module):
    """Loss that directly predicts selection rate quality."""
    
    def __init__(self, target_rate: float = 0.5):
        super().__init__()
        self.target = target_rate
    
    def forward(
        self,
        q_emb: torch.Tensor,
        d_embs: torch.Tensor,
        c_emb: torch.Tensor,
        rates: torch.Tensor
    ) -> torch.Tensor:
        """Loss based on distance from ideal selection rate."""
        n = d_embs.shape[0]
        if n < 1:
            return torch.tensor(0.0, device=q_emb.device, requires_grad=True)
        
        d_q_sims = F.cosine_similarity(d_embs.unsqueeze(0), q_emb.unsqueeze(0).unsqueeze(0))
        
        ideal_dist = torch.abs(rates - self.target)
        
        sorted_sims, indices = torch.sort(d_q_sims, descending=True)
        sorted_ideal = ideal_dist[indices]
        
        loss = torch.tensor(0.0, device=q_emb.device, requires_grad=True)
        for i in range(len(sorted_sims) - 1):
            if sorted_ideal[i] < sorted_ideal[i + 1]:
                if sorted_sims[i] < sorted_sims[i + 1]:
                    loss = loss + (sorted_sims[i + 1] - sorted_sims[i])
        
        return loss


class HybridLoss(nn.Module):
    """Combined loss for better optimization."""
    
    def __init__(self, contrastive_weight: float = 0.6):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.contrastive = ImprovedContrastiveLoss()
        self.quality = QualityPredictorLoss()
    
    def forward(
        self,
        q_emb: torch.Tensor,
        d_embs: torch.Tensor,
        c_emb: torch.Tensor,
        rates: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        c_loss = self.contrastive(q_emb, d_embs, c_emb, rates)
        q_loss = self.quality(q_emb, d_embs, c_emb, rates)
        
        total = self.contrastive_weight * c_loss + (1 - self.contrastive_weight) * q_loss
        
        return total, {
            'contrastive': c_loss.item() if torch.is_tensor(c_loss) else c_loss,
            'quality': q_loss.item() if torch.is_tensor(q_loss) else q_loss
        }


class FineTuner:
    """Fine-tuner for embedding models."""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loss_fn = HybridLoss()
        
        self._load_model()
    
    def _load_model(self):
        """Load model with LoRA."""
        from sentence_transformers import SentenceTransformer
        from peft import LoraConfig, get_peft_model, TaskType
        
        model_name = self.config['name']
        logger.info(f"Loading {model_name}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        base_model = self.model[0].auto_model
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["all-linear"],
            bias="none"
        )
        
        base_model = get_peft_model(base_model, peft_config)
        base_model.print_trainable_parameters()
        
        self.model[0].auto_model = base_model
        
        logger.info(f"Model {self.config['short_name']} loaded with LoRA")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts to embeddings."""
        prefix = self.config.get('prefix', '')
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def compute_batch_loss(self, batch: List[Dict]) -> torch.Tensor:
        """Compute loss for a batch."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_valid = 0
        
        for sample in batch:
            try:
                q_emb = self.encode([sample['question']])[0]
                d_embs = self.encode(sample['distractors'])
                c_emb = self.encode([sample['correct']])[0]
                rates = torch.tensor(
                    sample['selection_rates'],
                    device=self.device,
                    dtype=torch.float32
                )
                
                loss, _ = self.loss_fn(q_emb, d_embs, c_emb, rates)
                
                if not torch.isnan(loss) and loss.requires_grad:
                    total_loss = total_loss + loss
                    n_valid += 1
            except Exception as e:
                logger.warning(f"Error in batch: {e}")
                continue
        
        if n_valid > 0:
            return total_loss / n_valid
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def train(
        self,
        train_pairs: List[Dict],
        val_pairs: List[Dict],
        epochs: int = 10,
        patience: int = 3
    ) -> Dict:
        """Train the model."""
        lr = self.config.get('lr', 2e-5)
        batch_size = self.config.get('batch_size', 8)
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        n_batches = max(1, len(train_pairs) // batch_size)
        
        total_steps = n_batches * epochs
        warmup_steps = total_steps // 10
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_valid_batches = 0
            
            np.random.shuffle(train_pairs)
            
            progress = tqdm(
                range(0, len(train_pairs), batch_size),
                desc=f"Epoch {epoch+1}/{epochs}"
            )
            
            for i in progress:
                batch = train_pairs[i:i+batch_size]
                
                loss = self.compute_batch_loss(batch)
                
                if loss.requires_grad and not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    n_valid_batches += 1
                    progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = epoch_loss / max(1, n_valid_batches)
            history['train_loss'].append(avg_train_loss)
            
            val_loss = self._validate(val_pairs)
            history['val_loss'].append(val_loss)
            
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, lr={current_lr:.2e}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def _validate(self, val_pairs: List[Dict]) -> float:
        """Validate."""
        self.model.eval()
        total_loss = 0.0
        n_valid = 0
        batch_size = self.config.get('batch_size', 8)
        
        with torch.no_grad():
            for i in range(0, len(val_pairs), batch_size):
                batch = val_pairs[i:i+batch_size]
                
                loss = self.compute_batch_loss(batch)
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    n_valid += 1
        
        self.model.train()
        return total_loss / max(1, n_valid)
    
    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        output_dir = (
            Path(__file__).parent.parent / 'models' / 'finetuned' / 
            f"{self.config['short_name']}_{name}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))
        logger.info(f"Checkpoint saved to {output_dir}")
    
    def evaluate(self, pairs: List[Dict], n_folds: int = 5) -> Dict:
        """Evaluate with cross-validation."""
        self.model.eval()
        
        logger.info("Extracting features...")
        X_list = []
        y_list = []
        
        with torch.no_grad():
            for sample in tqdm(pairs, desc="Features"):
                try:
                    q_emb = self.encode([sample['question']])[0].cpu().numpy()
                    
                    for d_text, rate in zip(sample['distractors'], sample['selection_rates']):
                        d_emb = self.encode([d_text])[0].cpu().numpy()
                        c_emb = self.encode([sample['correct']])[0].cpu().numpy()
                        
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
                        y_list.append(rate)
                except Exception as e:
                    logger.warning(f"Error: {e}")
                    continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Feature matrix: {X.shape}")
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        all_metrics = {
            'MAE': [], 'RMSE': [], 'R2': [],
            'Pearson': [], 'Spearman': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            regressor = Ridge(alpha=1.0)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            
            all_metrics['MAE'].append(mean_absolute_error(y_test, y_pred))
            all_metrics['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            all_metrics['R2'].append(r2_score(y_test, y_pred))
            
            if len(y_test) > 1:
                p, _ = pearsonr(y_test, y_pred)
                s, _ = spearmanr(y_test, y_pred)
                all_metrics['Pearson'].append(p if not np.isnan(p) else 0.0)
                all_metrics['Spearman'].append(s if not np.isnan(s) else 0.0)
            else:
                all_metrics['Pearson'].append(0.0)
                all_metrics['Spearman'].append(0.0)
        
        results = {
            f'{k}_mean': float(np.mean(v)) for k, v in all_metrics.items()
        }
        results.update({
            f'{k}_std': float(np.std(v)) for k, v in all_metrics.items()
        })
        results['n_samples'] = len(y)
        results['n_folds'] = n_folds
        
        return results


def run_finetuning(config: Dict, all_pairs: List[Dict], output_dir: Path) -> Dict:
    """Run fine-tuning for a single model."""
    logger.info("=" * 70)
    logger.info(f"Fine-tuning: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    n_train = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]
    
    logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    tuner = FineTuner(config)
    
    logger.info("Zero-shot evaluation...")
    zero_shot_results = tuner.evaluate(all_pairs)
    logger.info(f"Zero-shot Pearson: {zero_shot_results['Pearson_mean']:.4f}")
    
    logger.info("Starting fine-tuning...")
    history = tuner.train(train_pairs, val_pairs, epochs=10, patience=3)
    
    logger.info("Post-tuning evaluation...")
    finetuned_results = tuner.evaluate(all_pairs)
    logger.info(f"Fine-tuned Pearson: {finetuned_results['Pearson_mean']:.4f}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    improvement = finetuned_results['Pearson_mean'] - zero_shot_results['Pearson_mean']
    
    result = {
        'model_name': config['name'],
        'short_name': config['short_name'],
        'description': config['description'],
        'config': {
            'lr': config.get('lr'),
            'batch_size': config.get('batch_size'),
            'dim': config.get('dim')
        },
        'duration_seconds': duration,
        'zero_shot': zero_shot_results,
        'finetuned': finetuned_results,
        'improvement': {
            'pearson': float(improvement),
            'percent': float(improvement / zero_shot_results['Pearson_mean'] * 100) if zero_shot_results['Pearson_mean'] != 0 else 0.0
        },
        'training_history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    result_file = output_dir / f"{config['short_name']}_finetune.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to {result_file}")
    
    logger.info("-" * 50)
    logger.info(f"Zero-shot Pearson: {zero_shot_results['Pearson_mean']:.4f}")
    logger.info(f"Fine-tuned Pearson: {finetuned_results['Pearson_mean']:.4f}")
    logger.info(f"Improvement: {improvement:+.4f} ({result['improvement']['percent']:+.1f}%)")
    logger.info(f"Duration: {duration:.1f}s")
    
    del tuner
    torch.cuda.empty_cache()
    
    return result


def main():
    """Main function."""
    logger.info("=" * 70)
    logger.info("FINE-TUNING ALL MODELS")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    
    project_dir = Path(__file__).parent.parent
    output_dir = project_dir / 'outputs' / 'results' / 'finetuned'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = load_all_data()
    all_pairs = prepare_pairs(samples)
    
    all_results = []
    
    for config in MODELS_TO_FINETUNE:
        try:
            result = run_finetuning(config, all_pairs, output_dir)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error fine-tuning {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'model_name': config['name'],
                'short_name': config['short_name'],
                'error': str(e)
            })
    
    summary_file = output_dir / 'all_finetuned_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_models': len(all_results),
            'n_samples': len(samples),
            'results': all_results
        }, f, indent=2, default=str)
    
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'Zero-shot':<12} {'Fine-tuned':<12} {'Improve':<12} {'%':<8}")
    logger.info("-" * 70)
    
    for r in all_results:
        if 'error' not in r:
            logger.info(
                f"{r['short_name']:<20} "
                f"{r['zero_shot']['Pearson_mean']:.4f}       "
                f"{r['finetuned']['Pearson_mean']:.4f}       "
                f"{r['improvement']['pearson']:+.4f}      "
                f"{r['improvement']['percent']:+.1f}%"
            )
        else:
            logger.info(f"{r['short_name']:<20} ERROR: {r['error'][:40]}")
    
    logger.info(f"\nCompleted: {datetime.now().isoformat()}")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
