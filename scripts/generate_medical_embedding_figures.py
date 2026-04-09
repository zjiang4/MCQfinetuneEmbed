#!/usr/bin/env python3
"""
Generate figures for medical embedding experiments for manuscript.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18

# Data
models_general = {
    'BGE-large (FT)': {'pearson': 0.653, 'mae': 0.0415, 'params': 335, 'type': 'General-Finetuned'},
    'BGE-base (FT)': {'pearson': 0.638, 'mae': 0.0400, 'params': 109, 'type': 'General-Finetuned'},
    'MPNet (FT)': {'pearson': 0.621, 'mae': 0.0399, 'params': 109, 'type': 'General-Finetuned'},
    'E5-large (FT)': {'pearson': 0.610, 'mae': 0.0410, 'params': 335, 'type': 'General-Finetuned'},
    'MiniLM (FT)': {'pearson': 0.612, 'mae': 0.0420, 'params': 22, 'type': 'General-Finetuned'},
}

models_medical = {
    'BioLORD-2023': {'pearson': 0.560, 'mae': 0.0445, 'params': 109, 'type': 'Medical-ZeroShot'},
    'MedCPT-Query': {'pearson': 0.560, 'mae': 0.0447, 'params': 109, 'type': 'Medical-ZeroShot'},
    'MedCPT-Article': {'pearson': 0.536, 'mae': 0.0460, 'params': 109, 'type': 'Medical-ZeroShot'},
    'MedEmbed-small': {'pearson': 0.534, 'mae': 0.0456, 'params': 33, 'type': 'Medical-ZeroShot'},
    'SapBERT-PubMed': {'pearson': 0.504, 'mae': 0.0467, 'params': 109, 'type': 'Medical-ZeroShot'},
}

models_baseline = {
    'BGE-base (BL)': {'pearson': 0.504, 'mae': 0.0510, 'params': 109, 'type': 'General-Baseline'},
}

output_dir = Path(__file__).parent.parent / 'paper' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)

# Figure 1: Comprehensive Model Comparison
print("Generating Figure 1: Comprehensive Model Comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Combine all models
all_models = {**models_baseline, **models_medical, **models_general}
model_names = list(all_models.keys())
pearson_scores = [all_models[m]['pearson'] for m in model_names]
mae_scores = [all_models[m]['mae'] for m in model_names]
types = [all_models[m]['type'] for m in model_names]

# Colors by type
color_map = {
    'General-Baseline': '#95a5a6',
    'Medical-ZeroShot': '#3498db',
    'General-Finetuned': '#e74c3c'
}
colors = [color_map[t] for t in types]

# Subplot 1: Pearson Correlation
y_pos = np.arange(len(model_names))
bars1 = ax1.barh(y_pos, pearson_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(model_names)
ax1.set_xlabel('Pearson Correlation Coefficient (r)', fontweight='bold')
ax1.set_title('A. Prediction Accuracy', fontweight='bold', pad=15)
ax1.set_xlim(0, 0.75)
ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, pearson_scores)):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Subplot 2: MAE
bars2 = ax2.barh(y_pos, mae_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(model_names)
ax2.set_xlabel('Mean Absolute Error (MAE)', fontweight='bold')
ax2.set_title('B. Prediction Error', fontweight='bold', pad=15)
ax2.set_xlim(0, 0.06)
ax2.axvline(x=0.045, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, mae_scores)):
    ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#95a5a6', edgecolor='black', label='General Baseline'),
    Patch(facecolor='#3498db', edgecolor='black', label='Medical Zero-Shot'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='General Fine-Tuned')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
           bbox_to_anchor=(0.5, 1.02), frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
fig.savefig(output_dir / 'figure1_model_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'figure1_model_comparison.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'figure1_model_comparison.png'}")

# Figure 2: Zero-shot vs Fine-tuned Performance
print("\nGenerating Figure 2: Zero-shot vs Fine-tuned Scatter...")
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
categories = {
    'General Baseline': models_baseline,
    'Medical Zero-Shot': models_medical,
    'General Fine-Tuned': models_general
}

markers = {'General Baseline': 'o', 'Medical Zero-Shot': 's', 'General Fine-Tuned': '^'}
colors_scatter = {'General Baseline': '#95a5a6', 'Medical Zero-Shot': '#3498db', 'General Fine-Tuned': '#e74c3c'}

for cat_name, cat_models in categories.items():
    for model_name, metrics in cat_models.items():
        ax.scatter(metrics['mae'], metrics['pearson'], 
                  s=metrics['params']*3,  # Size by parameters
                  c=colors_scatter[cat_name],
                  marker=markers[cat_name],
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=2,
                  label=cat_name if model_name == list(cat_models.keys())[0] else '')
        
        # Add model name annotation
        offset = (0.0008, 0.008)
        ax.annotate(model_name, 
                   xy=(metrics['mae'], metrics['pearson']),
                   xytext=(metrics['mae'] + offset[0], metrics['pearson'] + offset[1]),
                   fontsize=9,
                   alpha=0.8)

ax.set_xlabel('Mean Absolute Error (MAE) ↓', fontweight='bold', fontsize=14)
ax.set_ylabel('Pearson Correlation (r) ↑', fontweight='bold', fontsize=14)
ax.set_title('Performance Trade-off: Accuracy vs. Error', fontweight='bold', fontsize=16, pad=15)

# Add ideal region
ax.axhspan(0.60, 0.70, alpha=0.1, color='green', label='Target Region')
ax.axvspan(0.035, 0.045, alpha=0.1, color='green')

# Legend
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / 'figure2_performance_tradeoff.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'figure2_performance_tradeoff.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'figure2_performance_tradeoff.png'}")

# Figure 3: Computational Efficiency Trade-off
print("\nGenerating Figure 3: Computational Efficiency...")
fig, ax = plt.subplots(figsize=(12, 7))

# Data for efficiency comparison
approaches = [
    ('Medical\nZero-Shot\n(BioLORD-2023)', 0, 1.1, 0.560, 109),
    ('General\nBaseline\n(BGE-base)', 0, 1.1, 0.504, 109),
    ('Medical\nLoRA\n(MedEmbed-small)', 1.5, 1.1, 0.534, 33),
    ('General\nFine-Tuned\n(BGE-large)', 335, 3.2, 0.653, 335),
]

approach_names = [a[0] for a in approaches]
trainable_params = [a[1] for a in approaches]
gpu_memory = [a[2] for a in approaches]
pearson = [a[3] for a in approaches]
total_params = [a[4] for a in approaches]

# Create bars
x = np.arange(len(approach_names))
width = 0.25

# Normalize for visualization
trainable_norm = [p/350*100 if p > 0 else 5 for p in trainable_params]

bars1 = ax.bar(x - width, trainable_norm, width, label='Trainable Params (M)', 
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, [g*30 for g in gpu_memory], width, label='GPU Memory (GB×30)', 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, [p*100 for p in pearson], width, label='Pearson r (×100)', 
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Normalized Score', fontweight='bold', fontsize=14)
ax.set_xlabel('Deployment Approach', fontweight='bold', fontsize=14)
ax.set_title('Computational Efficiency vs. Performance Trade-off', fontweight='bold', fontsize=16, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(approach_names, fontsize=10)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', alpha=0.3)

# Add annotations
for i, (tp, gm, p) in enumerate(zip(trainable_params, gpu_memory, pearson)):
    if tp == 0:
        ax.text(i - width, trainable_norm[i] + 2, '0M', ha='center', fontsize=9, fontweight='bold')
    else:
        ax.text(i - width, trainable_norm[i] + 2, f'{tp:.1f}M', ha='center', fontsize=9, fontweight='bold')
    ax.text(i, gm*30 + 2, f'{gm:.1f}GB', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width, p*100 + 2, f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(output_dir / 'figure3_efficiency_tradeoff.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'figure3_efficiency_tradeoff.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'figure3_efficiency_tradeoff.png'}")

# Figure 4: Deployment Decision Tree
print("\nGenerating Figure 4: Deployment Decision Tree...")
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Box drawing function
def draw_box(ax, x, y, width, height, text, color, fontsize=11):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color,
                         linewidth=2, alpha=0.8)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
           fontsize=fontsize, fontweight='bold',
           wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, text=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))
    if text:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, text, fontsize=10, 
               style='italic', color='#2c3e50')

# Title
ax.text(5, 9.5, 'Model Selection Decision Framework', 
       ha='center', fontsize=18, fontweight='bold')

# Decision boxes
draw_box(ax, 5, 8, 3.5, 0.8, 'START:\nDistractor Quality Assessment', '#ecf0f1', fontsize=12)

# First decision
draw_box(ax, 5, 6.5, 3.2, 0.7, 'Training Data Available?', '#f39c12', fontsize=11)
draw_arrow(ax, 5, 7.6, 5, 6.85)

# No training data branch
draw_box(ax, 2.5, 5, 3, 0.7, 'Use Medical Embeddings\n(Zero-Shot)', '#3498db', fontsize=11)
draw_arrow(ax, 3.5, 6.5, 2.5, 5.35, 'No')

# GPU available for medical?
draw_box(ax, 2.5, 3.5, 2.8, 0.7, 'GPU Available?', '#f39c12', fontsize=11)
draw_arrow(ax, 2.5, 4.65, 2.5, 3.85)

# GPU yes for medical
draw_box(ax, 1.2, 2.2, 2.2, 0.9, '✓ BioLORD-2023\nr=0.560, 1.1GB\nBest accuracy', '#2ecc71', fontsize=10)
draw_arrow(ax, 2.5, 3.15, 1.2, 2.65, 'Yes')

# GPU no for medical
draw_box(ax, 3.8, 2.2, 2.2, 0.9, '✓ MedEmbed-small\nr=0.534, 1.1GB\nMost efficient', '#2ecc71', fontsize=10)
draw_arrow(ax, 2.5, 3.15, 3.8, 2.65, 'No')

# Training data available branch
draw_box(ax, 7.5, 5, 3, 0.7, 'Fine-Tune General\nEmbeddings', '#e74c3c', fontsize=11)
draw_arrow(ax, 6.5, 6.5, 7.5, 5.35, 'Yes')

# GPU available for fine-tuning?
draw_box(ax, 7.5, 3.5, 2.8, 0.7, 'High GPU Memory\n(≥16GB)?', '#f39c12', fontsize=11)
draw_arrow(ax, 7.5, 4.65, 7.5, 3.85)

# High GPU
draw_box(ax, 6.2, 2.2, 2.2, 0.9, '✓ BGE-large (FT)\nr=0.653, 3.2GB\nBest overall', '#2ecc71', fontsize=10)
draw_arrow(ax, 7.5, 3.15, 6.2, 2.65, 'Yes')

# Low GPU
draw_box(ax, 8.8, 2.2, 2.2, 0.9, '✓ MPNet (FT)\nr=0.621, 2.8GB\nGood trade-off', '#2ecc71', fontsize=10)
draw_arrow(ax, 7.5, 3.15, 8.8, 2.65, 'No')

# Add legend
legend_y = 1.2
ax.text(1, legend_y, 'Color Legend:', fontsize=11, fontweight='bold')
draw_box(ax, 2, legend_y - 0.3, 1.2, 0.4, 'Start', '#ecf0f1', fontsize=9)
draw_box(ax, 3.5, legend_y - 0.3, 1.2, 0.4, 'Decision', '#f39c12', fontsize=9)
draw_box(ax, 5, legend_y - 0.3, 1.5, 0.4, 'Medical', '#3498db', fontsize=9)
draw_box(ax, 6.7, legend_y - 0.3, 1.5, 0.4, 'Fine-Tuned', '#e74c3c', fontsize=9)
draw_box(ax, 8.4, legend_y - 0.3, 1.2, 0.4, 'Recommended', '#2ecc71', fontsize=9)

plt.tight_layout()
fig.savefig(output_dir / 'figure4_decision_tree.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'figure4_decision_tree.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'figure4_decision_tree.png'}")

# Figure 5: Medical vs General Performance Comparison
print("\nGenerating Figure 5: Medical vs General Comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['General\nBaseline', 'Medical\nZero-Shot', 'General\nFine-Tuned']
pearson_values = [0.504, 0.560, 0.653]
mae_values = [0.0510, 0.0445, 0.0415]
colors_bar = ['#95a5a6', '#3498db', '#e74c3c']

x = np.arange(len(categories))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, pearson_values, width, label='Pearson r',
               color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)

# Add percentage improvements
improvements = ['', '+11.1%', '+16.6%']
for i, (bar, imp) in enumerate(zip(bars1, improvements)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{height:.3f}\n{imp}',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Pearson Correlation Coefficient', fontweight='bold', fontsize=14)
ax.set_xlabel('Model Category', fontweight='bold', fontsize=14)
ax.set_title('Performance Progression: Baseline → Medical → Fine-Tuned', 
            fontweight='bold', fontsize=16, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 0.75)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random baseline')
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / 'figure5_category_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(output_dir / 'figure5_category_comparison.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'figure5_category_comparison.png'}")

print("\n" + "="*70)
print("✅ All figures generated successfully!")
print("="*70)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  - figure1_model_comparison.png/pdf")
print("  - figure2_performance_tradeoff.png/pdf")
print("  - figure3_efficiency_tradeoff.png/pdf")
print("  - figure4_decision_tree.png/pdf")
print("  - figure5_category_comparison.png/pdf")
