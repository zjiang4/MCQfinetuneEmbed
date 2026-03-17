#!/usr/bin/env python3
"""
Multi-Model Baseline Experiments for MCQ Distractor Quality Prediction.

Tests 6 embedding models on the full dataset:
1. BAAI/bge-large-en-v1.5 - English SOTA
2. sentence-transformers/all-mpnet-base-v2 - High quality general
3. intfloat/e5-large-v2 - Microsoft E5
4. sentence-transformers/all-MiniLM-L6-v2 - Lightweight efficient
5. BAAI/bge-m3 - Multilingual (good English)
6. emilyalsentzer/Bio_ClinicalBERT - Medical domain
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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

import torch
import torch.nn.functional as F

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MODEL_CONFIGS = [
    {
        'name': 'BAAI/bge-large-en-v1.5',
        'short_name': 'bge-large-en',
        'dim': 1024,
        'type': 'sentence_transformer',
        'description': 'BGE Large English v1.5 - SOTA English embedding'
    },
    {
        'name': 'sentence-transformers/all-mpnet-base-v2',
        'short_name': 'all-mpnet-base',
        'dim': 768,
        'type': 'sentence_transformer',
        'description': 'All-MPNet Base v2 - High quality general purpose'
    },
    {
        'name': 'intfloat/e5-large-v2',
        'short_name': 'e5-large-v2',
        'dim': 1024,
        'type': 'e5',
        'description': 'E5 Large v2 - Microsoft embedding model'
    },
    {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'short_name': 'all-MiniLM',
        'dim': 384,
        'type': 'sentence_transformer',
        'description': 'MiniLM L6 v2 - Lightweight efficient model'
    },
    {
        'name': 'BAAI/bge-base-en-v1.5',
        'short_name': 'bge-base-en',
        'dim': 768,
        'type': 'sentence_transformer',
        'description': 'BGE Base English v1.5 - Smaller BGE variant'
    },
    {
        'name': 'dmis-lab/biobert-base-cased-v1.1',
        'short_name': 'biobert-base',
        'dim': 768,
        'type': 'bert',
        'description': 'BioBERT Base - Biomedical domain specific'
    },
]


def load_raw_data(data_dir: str) -> List[Dict]:
    """Load all data from processed dataset."""
    all_samples = []
    data_path = Path(data_dir)
    
    processed_file = data_path.parent / 'data' / 'processed' / 'full_dataset.json'
    
    if processed_file.exists():
        logger.info(f"Loading processed data from {processed_file}...")
        with open(processed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_samples = data.get('samples', data)
        logger.info(f"Total samples loaded: {len(all_samples)}")
        return all_samples
    
    for txt_file in sorted(data_path.glob('*.txt')):
        logger.info(f"Loading {txt_file.name}...")
        
        try:
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']:
                try:
                    with open(txt_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if content is None:
                logger.error(f"Could not read {txt_file.name} with any encoding")
                continue
            
            content = content.replace('\ufeff', '').strip()
            if not content:
                logger.warning(f"Empty file: {txt_file.name}")
                continue
            
            data = json.loads(content)
            questions = data.get('questions', [])
            
            for q in questions:
                options = []
                correct_answer = None
                
                for opt in q.get('options', []):
                    if 'correctOption' in opt:
                        correct_answer = opt['correctOption']
                        options.append({
                            'text': opt['correctOption'],
                            'is_correct': True,
                            'selection_rate': parse_rate(opt.get('selectionRate', '0%'))
                        })
                    elif 'wrongOption' in opt:
                        options.append({
                            'text': opt['wrongOption'],
                            'is_correct': False,
                            'selection_rate': parse_rate(opt.get('selectionRate', '0%'))
                        })
                
                if correct_answer and len([o for o in options if not o['is_correct']]) >= 1:
                    all_samples.append({
                        'id': q.get('id', f"{txt_file.stem}_{len(all_samples)}"),
                        'question': q.get('question', ''),
                        'content_area': q.get('contentArea', txt_file.stem),
                        'options': options,
                        'explanation': q.get('explanation', ''),
                        'key_learning_points': q.get('keyLearningPoints', '')
                    })
        
        except Exception as e:
            logger.error(f"Error loading {txt_file.name}: {e}")
            continue
    
    logger.info(f"Total samples loaded: {len(all_samples)}")
    return all_samples


def parse_rate(rate_str: str) -> float:
    """Parse selection rate string to float."""
    try:
        if isinstance(rate_str, (int, float)):
            return float(rate_str)
        rate_str = str(rate_str).strip().replace('%', '')
        return float(rate_str) / 100.0
    except:
        return 0.0


class EmbeddingModel:
    """Wrapper for different embedding model types."""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.short_name = config['short_name']
        self._load_model()
    
    def _load_model(self):
        """Load model based on type."""
        model_name = self.config['name']
        model_type = self.config['type']
        
        logger.info(f"Loading {model_name} (type: {model_type})...")
        
        try:
            if model_type == 'sentence_transformer':
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device=self.device)
            
            elif model_type == 'e5':
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device=self.device)
            
            elif model_type == 'bert':
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            
            logger.info(f"Model {self.short_name} loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        model_type = self.config['type']
        
        if model_type == 'sentence_transformer':
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        
        elif model_type == 'e5':
            prefixed_texts = [f"query: {t}" for t in texts]
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        
        elif model_type == 'bert':
            all_embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    
                    encoded = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    )
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    
                    outputs = self.model(**encoded)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.append(embeddings.cpu().numpy())
            
            return np.vstack(all_embeddings)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def extract_features(
    model: EmbeddingModel,
    samples: List[Dict],
    batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from samples."""
    X_list = []
    y_list = []
    
    logger.info(f"Extracting features with {model.short_name}...")
    
    for sample in tqdm(samples, desc=f"Features [{model.short_name}]"):
        try:
            q_emb = model.encode([sample['question']], batch_size=1)[0]
            
            for opt in sample['options']:
                if opt['is_correct']:
                    continue
                
                d_emb = model.encode([opt['text']], batch_size=1)[0]
                
                correct_opt = next((o for o in sample['options'] if o['is_correct']), None)
                if correct_opt:
                    c_emb = model.encode([correct_opt['text']], batch_size=1)[0]
                else:
                    c_emb = np.zeros_like(d_emb)
                
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
        
        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            continue
    
    return np.array(X_list), np.array(y_list)


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    n_splits: int = 5
) -> Dict:
    """Evaluate model with cross-validation."""
    from sklearn.model_selection import KFold
    
    results = {
        'fold_results': [],
        'mean_metrics': {}
    }
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_maes = []
    all_pearsons = []
    all_spearmans = []
    all_r2s = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        regressor = Ridge(alpha=1.0)
        regressor.fit(X_train, y_train)
        
        y_pred = regressor.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        if len(y_test) > 1:
            pearson, _ = pearsonr(y_test, y_pred)
            spearman, _ = spearmanr(y_test, y_pred)
        else:
            pearson, spearman = 0.0, 0.0
        
        if np.isnan(pearson):
            pearson = 0.0
        if np.isnan(spearman):
            spearman = 0.0
        
        fold_result = {
            'fold': fold + 1,
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Pearson': float(pearson),
            'Spearman': float(spearman),
            'n_test': len(y_test)
        }
        results['fold_results'].append(fold_result)
        
        all_maes.append(mae)
        all_pearsons.append(pearson)
        all_spearmans.append(spearman)
        all_r2s.append(r2)
    
    results['mean_metrics'] = {
        'MAE': float(np.mean(all_maes)),
        'MAE_std': float(np.std(all_maes)),
        'Pearson': float(np.mean(all_pearsons)),
        'Pearson_std': float(np.std(all_pearsons)),
        'Spearman': float(np.mean(all_spearmans)),
        'Spearman_std': float(np.std(all_spearmans)),
        'R2': float(np.mean(all_r2s)),
        'R2_std': float(np.std(all_r2s))
    }
    
    return results


def run_single_model(
    config: Dict,
    samples: List[Dict],
    output_dir: Path
) -> Dict:
    """Run baseline experiment for a single model."""
    logger.info("=" * 70)
    logger.info(f"Testing: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    try:
        model = EmbeddingModel(config, device='cuda')
        
        X, y = extract_features(model, samples)
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        
        results = evaluate_model(X, y, n_splits=5)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = {
            'model_name': config['name'],
            'short_name': config['short_name'],
            'description': config['description'],
            'embedding_dim': config['dim'],
            'feature_dim': X.shape[1],
            'n_samples': len(y),
            'duration_seconds': duration,
            'mean_metrics': results['mean_metrics'],
            'fold_results': results['fold_results']
        }
        
        output_file = output_dir / f"{config['short_name']}_baseline.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
        
        logger.info(f"Mean MAE: {results['mean_metrics']['MAE']:.4f} ± {results['mean_metrics']['MAE_std']:.4f}")
        logger.info(f"Mean Pearson: {results['mean_metrics']['Pearson']:.4f} ± {results['mean_metrics']['Pearson_std']:.4f}")
        logger.info(f"Duration: {duration:.1f}s")
        
        del model
        torch.cuda.empty_cache()
        
        return result
    
    except Exception as e:
        logger.error(f"Error testing {config['name']}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'model_name': config['name'],
            'short_name': config['short_name'],
            'error': str(e)
        }


def main():
    """Run all baseline experiments."""
    logger.info("=" * 70)
    logger.info("MULTI-MODEL BASELINE EXPERIMENTS")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / '题目'
    output_dir = project_dir / 'outputs' / 'results' / 'baselines'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = load_raw_data(str(data_dir))
    logger.info(f"Total samples: {len(samples)}")
    
    all_results = []
    
    for config in MODEL_CONFIGS:
        result = run_single_model(config, samples, output_dir)
        all_results.append(result)
    
    summary_file = output_dir / 'all_baselines_summary.json'
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
    logger.info(f"{'Model':<25} {'MAE':<12} {'Pearson':<12} {'Spearman':<12}")
    logger.info("-" * 65)
    
    for result in all_results:
        if 'error' not in result:
            m = result['mean_metrics']
            logger.info(
                f"{result['short_name']:<25} "
                f"{m['MAE']:.4f}±{m['MAE_std']:.3f}  "
                f"{m['Pearson']:.4f}±{m['Pearson_std']:.3f}  "
                f"{m['Spearman']:.4f}±{m['Spearman_std']:.3f}"
            )
        else:
            logger.info(f"{result['short_name']:<25} ERROR: {result['error'][:30]}")
    
    logger.info(f"\nCompleted: {datetime.now().isoformat()}")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
