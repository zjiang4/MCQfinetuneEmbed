#!/usr/bin/env python3
"""
统计显著性检验

比较:
1. 微调前后模型性能
2. 不同模型之间的性能差异
3. 与baseline的对比

使用方法:
- Paired t-test: 比较同一数据上两个模型的预测
- Wilcoxon signed-rank test: 非参数检验
- Bootstrap confidence intervals
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 同时保存到文件
output_dir = Path(__file__).parent.parent / 'outputs' / 'results' / 'statistics'
output_dir.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(output_dir / 'statistical_tests.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


def load_data():
    """加载数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'full_dataset.json'
    with open(data_path) as f:
        data = json.load(f)
    return data['samples']


def prepare_pairs(samples: List[Dict]) -> List[Dict]:
    """准备训练对"""
    pairs = []
    for s in samples:
        q = s['question']
        options = s['options']
        correct = next((o['text'] for o in options if o['is_correct']), None)
        distractors = [(o['text'], o['selection_rate']) for o in options if not o['is_correct']]
        
        if correct and len(distractors) >= 2:
            pairs.append({
                'question': q,
                'correct': correct,
                'distractors': [d[0] for d in distractors],
                'rates': [d[1] for d in distractors]
            })
    return pairs


def get_predictions_and_targets(model, pairs: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取模型预测和真实值
    
    Returns:
        predictions: 模型预测的选择率
        targets: 真实选择率
    """
    X_list = []
    y_list = []
    
    for p in tqdm(pairs, desc='Getting predictions', leave=False):
        q_emb = model.encode(p['question'], normalize_embeddings=True)
        for d, rate in zip(p['distractors'], p['rates']):
            d_emb = model.encode(d, normalize_embeddings=True)
            feat = np.concatenate([
                d_emb,
                q_emb * d_emb,
                [np.dot(q_emb, d_emb)]
            ])
            X_list.append(feat)
            y_list.append(rate)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # 使用5折交叉验证获得out-of-fold预测
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.zeros(len(y))
    
    for train_idx, test_idx in kf.split(X):
        reg = Ridge(alpha=1.0)
        reg.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = reg.predict(X[test_idx])
    
    return predictions, y


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """计算各种评估指标"""
    # Pearson correlation
    pearson, pearson_p = pearsonr(targets, predictions)
    
    # Spearman correlation
    spearman, spearman_p = spearmanr(targets, predictions)
    
    # MAE
    mae = np.mean(np.abs(targets - predictions))
    
    # RMSE
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    
    # R2
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'pearson': pearson,
        'pearson_p': pearson_p,
        'spearman': spearman,
        'spearman_p': spearman_p,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_samples': len(targets)
    }


def paired_t_test(pred1: np.ndarray, pred2: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Paired t-test比较两个模型
    
    使用Fisher z-transform对correlation进行转换后再比较
    """
    # 方法1: 直接比较correlation的差异
    r1, _ = pearsonr(targets, pred1)
    r2, _ = pearsonr(targets, pred2)
    
    # Fisher z-transform
    n = len(targets)
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    
    # 标准误
    se = np.sqrt(2 / (n - 3))
    
    # z统计量
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # 方法2: 比较预测误差
    errors1 = np.abs(targets - pred1)
    errors2 = np.abs(targets - pred2)
    
    t_stat, t_p = ttest_rel(errors1, errors2)
    
    return {
        'model1_r': r1,
        'model2_r': r2,
        'z_statistic': z_stat,
        'z_p_value': p_value,
        't_statistic': t_stat,
        't_p_value': t_p,
        'significant_z': p_value < 0.05,
        'significant_t': t_p < 0.05
    }


def wilcoxon_test(pred1: np.ndarray, pred2: np.ndarray, targets: np.ndarray) -> Dict:
    """Wilcoxon signed-rank test (非参数检验)"""
    # 比较绝对误差
    errors1 = np.abs(targets - pred1)
    errors2 = np.abs(targets - pred2)
    
    # Wilcoxon test
    statistic, p_value = wilcoxon(errors1, errors2, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def bootstrap_ci(predictions: np.ndarray, targets: np.ndarray, 
                 n_bootstrap: int = 1000, ci: float = 0.95) -> Dict:
    """
    Bootstrap置信区间
    """
    n_samples = len(targets)
    pearsons = []
    
    np.random.seed(42)
    for _ in tqdm(range(n_bootstrap), desc='Bootstrap', leave=False):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        r, _ = pearsonr(targets[indices], predictions[indices])
        pearsons.append(r)
    
    pearsons = np.array(pearsons)
    
    lower = np.percentile(pearsons, (1 - ci) / 2 * 100)
    upper = np.percentile(pearsons, (1 + ci) / 2 * 100)
    
    return {
        'mean': np.mean(pearsons),
        'std': np.std(pearsons),
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence_level': ci
    }


def run_statistical_tests():
    """运行所有统计检验"""
    logger.info("=" * 70)
    logger.info("统计显著性检验")
    logger.info("=" * 70)
    logger.info(f"时间: {datetime.now().isoformat()}")
    
    # 加载数据
    samples = load_data()
    pairs = prepare_pairs(samples)
    logger.info(f"数据: {len(samples)} 样本, {len(pairs)} 对")
    
    # 模型配置
    models_config = [
        {
            'name': 'E5-large-v2 (Zero-shot)',
            'path': 'intfloat/e5-large-v2',
            'type': 'zero_shot'
        },
        {
            'name': 'E5-large-v2 (Fine-tuned)',
            'path': 'models/finetuned/e5',
            'type': 'finetuned'
        },
        {
            'name': 'BGE-base-en (Zero-shot)',
            'path': 'BAAI/bge-base-en-v1.5',
            'type': 'zero_shot'
        },
        {
            'name': 'BGE-base-en (Fine-tuned)',
            'path': 'models/finetuned/bge-base',
            'type': 'finetuned'
        },
        {
            'name': 'BGE-large-en (Zero-shot)',
            'path': 'BAAI/bge-large-en-v1.5',
            'type': 'zero_shot'
        },
        {
            'name': 'BGE-large-en (Fine-tuned)',
            'path': 'models/finetuned/bge-large',
            'type': 'finetuned'
        },
    ]
    
    # 获取每个模型的预测
    from sentence_transformers import SentenceTransformer
    
    all_results = {}
    all_predictions = {}
    
    for config in models_config:
        logger.info(f"\n加载模型: {config['name']}")
        
        try:
            if config['type'] == 'finetuned':
                model_path = Path(__file__).parent.parent / config['path']
                model = SentenceTransformer(str(model_path), device='cuda')
            else:
                model = SentenceTransformer(config['path'], device='cuda')
            
            predictions, targets = get_predictions_and_targets(model, pairs)
            metrics = compute_metrics(predictions, targets)
            
            logger.info(f"Pearson: {metrics['pearson']:.4f} (p={metrics['pearson_p']:.2e})")
            logger.info(f"Spearman: {metrics['spearman']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            
            # Bootstrap CI
            logger.info("计算Bootstrap置信区间...")
            bootstrap = bootstrap_ci(predictions, targets, n_bootstrap=500)
            logger.info(f"95% CI: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")
            
            all_results[config['name']] = {
                'metrics': metrics,
                'bootstrap': bootstrap
            }
            all_predictions[config['name']] = predictions
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            continue
    
    # 目标值相同
    targets = np.array([r for p in pairs for r in p['rates']])
    
    # 比较: 微调前后
    logger.info("\n" + "=" * 70)
    logger.info("微调前后对比 (Paired t-test)")
    logger.info("=" * 70)
    
    comparisons = [
        ('E5-large-v2 (Zero-shot)', 'E5-large-v2 (Fine-tuned)'),
        ('BGE-base-en (Zero-shot)', 'BGE-base-en (Fine-tuned)'),
        ('BGE-large-en (Zero-shot)', 'BGE-large-en (Fine-tuned)'),
    ]
    
    t_test_results = {}
    for model1, model2 in comparisons:
        if model1 in all_predictions and model2 in all_predictions:
            logger.info(f"\n比较: {model1} vs {model2}")
            result = paired_t_test(all_predictions[model1], all_predictions[model2], targets)
            logger.info(f"  Model1 r: {result['model1_r']:.4f}")
            logger.info(f"  Model2 r: {result['model2_r']:.4f}")
            logger.info(f"  Z-statistic: {result['z_statistic']:.4f}")
            logger.info(f"  Z p-value: {result['z_p_value']:.2e}")
            logger.info(f"  显著 (p<0.05): {result['significant_z']}")
            
            t_test_results[f"{model1}_vs_{model2}"] = result
    
    # Wilcoxon检验
    logger.info("\n" + "=" * 70)
    logger.info("Wilcoxon Signed-Rank Test")
    logger.info("=" * 70)
    
    wilcoxon_results = {}
    for model1, model2 in comparisons:
        if model1 in all_predictions and model2 in all_predictions:
            logger.info(f"\n比较: {model1} vs {model2}")
            result = wilcoxon_test(all_predictions[model1], all_predictions[model2], targets)
            logger.info(f"  Statistic: {result['statistic']:.1f}")
            logger.info(f"  p-value: {result['p_value']:.2e}")
            logger.info(f"  显著: {result['significant']}")
            
            wilcoxon_results[f"{model1}_vs_{model2}"] = result
    
    # 保存结果
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(targets),
        'model_results': all_results,
        'paired_t_tests': t_test_results,
        'wilcoxon_tests': wilcoxon_results
    }
    
    output_file = output_dir / 'statistical_tests_results.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info(f"\n结果保存到: {output_file}")
    
    # 汇总
    logger.info("\n" + "=" * 70)
    logger.info("汇总")
    logger.info("=" * 70)
    
    logger.info("\n各模型性能:")
    logger.info(f"{'模型':<35} {'Pearson':<12} {'95% CI':<20} {'MAE':<10}")
    logger.info("-" * 80)
    for name, res in all_results.items():
        m = res['metrics']
        b = res['bootstrap']
        logger.info(f"{name:<35} {m['pearson']:.4f}       [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]      {m['mae']:.4f}")
    
    logger.info("\n微调效果显著性:")
    for comp, res in t_test_results.items():
        logger.info(f"  {comp}: {'显著' if res['significant_z'] else '不显著'} (p={res['z_p_value']:.2e})")
    
    return final_results


if __name__ == "__main__":
    run_statistical_tests()
