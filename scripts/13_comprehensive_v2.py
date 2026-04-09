#!/usr/bin/env python3
"""
Simplified comprehensive experiments with better logging
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(PROJECT_DIR / 'logs' / '13_comprehensive_v2.log'))
    ],
    force=True
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data' / 'processed'
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'results' / 'comprehensive_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'


def find_model_path(model_name: str) -> str:
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    patterns = {
        'bge-large': 'models--BAAI--bge-large-en-v1.5',
        'bge-base': 'models--BAAI--bge-base-en-v1.5',
        'mpnet': 'models--sentence-transformers--all-mpnet-base-v2',
        'minilm': 'models--sentence-transformers--all-MiniLM-L6-v2',
        'e5-large': 'models--intfloat--e5-large-v2',
    }
    pattern = patterns.get(model_name)
    if not pattern:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_dir = cache_dir / pattern
    snapshots_dir = model_dir / 'snapshots'
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[0])
    return str(model_dir)


def load_data(split: str) -> List[Dict]:
    with open(DATA_DIR / f'{split}.json') as f:
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
        q, opts = s['question'], s['options']
        for opt in opts:
            if not opt['is_correct']:
                texts.append(f"Question: {q} Option: {opt['text']}")
                rates.append(opt['selection_rate'])
    return texts, rates


class DistractorPredictor(nn.Module):
    def __init__(self, model_path: str, hidden_dim: int = 256, freeze_encoder: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        if freeze_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(outputs.last_hidden_state[:, 0, :]).squeeze(-1)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()
    
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        huber = self.huber(pred, target)
        return mse + 0.5 * huber


LOSSES = {
    'mse': nn.MSELoss(),
    'mae': nn.L1Loss(),
    'huber': nn.SmoothL1Loss(),
    'combined': CombinedLoss(),
}


def run_baseline(model_name: str, test_data: List[Dict]) -> Dict:
    logger.info(f"Running baseline for {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = find_model_path(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    bert = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    
    texts, labels = create_training_data(test_data)
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tokenizer(texts[i:i+64], padding=True, truncation=True, max_length=512, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = bert(**enc)
            embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    
    embeddings, labels = np.vstack(embeddings), np.array(labels)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    pearsons = []
    for train_idx, test_idx in kfold.split(embeddings):
        ridge = Ridge(alpha=1.0).fit(embeddings[train_idx], labels[train_idx])
        r, _ = pearsonr(labels[test_idx], ridge.predict(embeddings[test_idx]))
        pearsons.append(r)
    
    result = {'model': model_name, 'Pearson_mean': float(np.mean(pearsons)), 'Pearson_std': float(np.std(pearsons))}
    logger.info(f"Baseline {model_name}: Pearson={result['Pearson_mean']:.4f}")
    return result


def run_experiment(model_name: str, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict],
                   loss_name: str, freeze: bool, lr: float, epochs: int = 5, batch_size: int = 32) -> Dict:
    exp_name = f"{model_name}_{loss_name}_{'frozen' if freeze else 'full'}_lr{lr}"
    logger.info(f"\nExperiment: {exp_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = find_model_path(model_name)
    
    train_texts, train_labels = create_training_data(train_data)
    val_texts, val_labels = create_training_data(val_data)
    test_texts, test_labels = create_training_data(test_data)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc, self.labels = enc, torch.tensor(labels, dtype=torch.float32)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()} | {'labels': self.labels[i]}
    
    train_loader = torch.utils.data.DataLoader(Dataset(train_enc, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(Dataset(val_enc, val_labels), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(Dataset(test_enc, test_labels), batch_size=batch_size)
    
    model = DistractorPredictor(model_path, freeze_encoder=freeze).to(device)
    criterion = LOSSES[loss_name]
    
    params = model.regressor.parameters() if freeze else model.parameters()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable: {n_trainable:,} params")
    
    best_pearson, patience_cnt, best_state = -1, 0, None
    
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for batch in train_loader:
            ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        
        model.eval()
        preds, lbls = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                preds.extend(model(ids, mask).cpu().numpy())
                lbls.extend(batch['labels'].numpy())
        
        val_metrics = compute_metrics(np.array(lbls), np.array(preds))
        logger.info(f"Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}, val_pearson={val_metrics['Pearson_r']:.4f}")
        
        if val_metrics['Pearson_r'] > best_pearson:
            best_pearson = val_metrics['Pearson_r']
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        
        if patience_cnt >= 2:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            preds.extend(model(ids, mask).cpu().numpy())
            lbls.extend(batch['labels'].numpy())
    
    test_metrics = compute_metrics(np.array(lbls), np.array(preds))
    logger.info(f"Test: Pearson={test_metrics['Pearson_r']:.4f}, MAE={test_metrics['MAE']:.4f}")
    
    result = {
        'experiment_name': exp_name,
        'model': model_name,
        'loss': loss_name,
        'freeze': freeze,
        'lr': lr,
        'trainable_params': n_trainable,
        'best_val_pearson': float(best_pearson),
        'test_metrics': test_metrics,
        'epochs_trained': epoch + 1,
    }
    
    with open(OUTPUT_DIR / f'{exp_name}.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def main():
    logger.info("="*80)
    logger.info("COMPREHENSIVE EXPERIMENTS V2")
    logger.info(f"Started: {datetime.now()}")
    
    train_data = load_data('train')
    val_data = load_data('val')
    test_data = load_data('test')
    logger.info(f"Data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    all_results = []
    
    models = ['bge-large', 'mpnet']
    losses = ['mse', 'mae', 'huber', 'combined']
    freezes = [True, False]
    lrs = [1e-5, 2e-5, 5e-5]
    
    for model_name in models:
        baseline = run_baseline(model_name, test_data)
        all_results.append(baseline)
        
        for loss_name in losses:
            for freeze in freezes:
                for lr in lrs:
                    try:
                        result = run_experiment(
                            model_name, train_data, val_data, test_data,
                            loss_name, freeze, lr
                        )
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Failed: {model_name}_{loss_name}_{'frozen' if freeze else 'full'}_lr{lr}: {e}")
                        import traceback
                        traceback.print_exc()
    
    with open(OUTPUT_DIR / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    successful = [r for r in all_results if 'error' not in r and 'test_metrics' in r]
    sorted_results = sorted(successful, key=lambda x: x.get('test_metrics', {}).get('Pearson_r', 0), reverse=True)
    
    logger.info("\n" + "="*80)
    logger.info("TOP 10 RESULTS")
    for i, r in enumerate(sorted_results[:10], 1):
        name = r.get('experiment_name', r.get('model', '?'))
        pearson = r.get('test_metrics', {}).get('Pearson_r', r.get('Pearson_mean', 0))
        logger.info(f"{i}. {name}: {pearson:.4f}")
    
    logger.info(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
