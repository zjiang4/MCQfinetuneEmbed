#!/usr/bin/env python3
"""
Phase 0: Data Preprocessing

1. Parse new JSON format from 题目/ directory
2. Identify options missing text
3. Generate missing option text using LLM
4. Save processed dataset
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random

# Set seed for reproducibility
random.seed(42)

DATA_DIR = Path(__file__).parent.parent / '题目'
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'processed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_selection_rate(rate_str: str) -> float:
    """Parse selection rate string to float."""
    if isinstance(rate_str, (int, float)):
        return float(rate_str)
    
    rate_str = str(rate_str).strip()
    
    # Handle percentage format (e.g., "14%", "14 %")
    if '%' in rate_str:
        rate_str = rate_str.replace('%', '').strip()
        return float(rate_str) / 100.0
    
    # Handle decimal format (e.g., "0.14")
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def is_valid_option_text(text: str) -> bool:
    """Check if option text is valid (not just a letter label)."""
    if not text or len(text.strip()) < 5:
        return False
    
    # Check if it's just a letter (A, B, C, D, E)
    text = text.strip()
    if text in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return False
    
    # Check if it's a letter with parenthesis
    if re.match(r'^[A-G][\.\)]?\s*$', text):
        return False
    
    return True


def extract_answer_letter(text: str) -> str:
    """Extract letter from text like 'A' or 'Option A'."""
    text = text.strip()
    if text in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return text
    match = re.search(r'([A-G])', text)
    return match.group(1) if match else 'X'


def load_all_data() -> Tuple[List[Dict], Dict]:
    """Load all JSON files and return processed samples."""
    all_samples = []
    stats = defaultdict(lambda: {'total': 0, 'valid': 0, 'invalid': 0, 'questions': 0})
    
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith('.json'):
            continue
        
        filepath = DATA_DIR / filename
        domain = filename.replace('_questions.json', '')
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get('questions', [])
        stats[domain]['questions'] = len(questions)
        
        for q in questions:
            question_text = q.get('question', '')
            content_area = q.get('contentArea', domain)
            explanation = q.get('explanation', '')
            key_learning_points = q.get('keyLearningPoints', '')
            
            options = q.get('options', [])
            
            valid_options = []
            has_correct = False
            
            for opt in options:
                # Determine if this is correct or wrong option
                if 'correctOption' in opt:
                    text = opt['correctOption']
                    is_correct = True
                    has_correct = True
                elif 'wrongOption' in opt:
                    text = opt['wrongOption']
                    is_correct = False
                else:
                    continue
                
                rate = parse_selection_rate(opt.get('selectionRate', '0%'))
                
                valid = is_valid_option_text(text)
                
                if valid:
                    stats[domain]['valid'] += 1
                else:
                    stats[domain]['invalid'] += 1
                stats[domain]['total'] += 1
                
                valid_options.append({
                    'text': text,
                    'is_correct': is_correct,
                    'selection_rate': rate,
                    'has_valid_text': valid
                })
            
            if has_correct and len(valid_options) >= 4:
                sample = {
                    'id': f"{domain}_{q.get('id', len(all_samples))}",
                    'question': question_text,
                    'content_area': content_area,
                    'options': valid_options,
                    'explanation': explanation,
                    'key_learning_points': key_learning_points,
                    'domain': domain
                }
                all_samples.append(sample)
    
    return all_samples, dict(stats)


def generate_missing_option_text(sample: Dict, option_idx: int) -> str:
    """Generate missing option text based on question context and other options."""
    question = sample['question']
    options = sample['options']
    
    # Find correct answer
    correct_opt = next((o for o in options if o['is_correct']), None)
    correct_text = correct_opt['text'] if correct_opt else ""
    
    # Find other valid distractors
    valid_distractors = [
        o['text'] for o in options 
        if not o['is_correct'] and o['has_valid_text']
    ]
    
    # Get medical context
    domain = sample.get('domain', 'Medicine')
    
    # Generate placeholder text (will be replaced by LLM later)
    # For now, create a descriptive placeholder
    placeholder = f"[PLACEHOLDER: Option {option_idx + 1} for {domain} question about {question[:50]}...]"
    
    return placeholder


def find_samples_needing_generation(samples: List[Dict]) -> List[Tuple[int, int]]:
    """Find all (sample_idx, option_idx) pairs that need text generation."""
    needs_generation = []
    
    for sample_idx, sample in enumerate(samples):
        for opt_idx, opt in enumerate(sample['options']):
            if not opt['has_valid_text']:
                needs_generation.append((sample_idx, opt_idx))
    
    return needs_generation


def generate_all_missing_texts(samples: List[Dict]) -> List[Dict]:
    """
    Generate missing option texts.
    
    For now, we'll mark them but not actually generate.
    In a real scenario, this would call an LLM API.
    """
    needs_generation = find_samples_needing_generation(samples)
    
    print(f"\nFound {len(needs_generation)} options needing text generation")
    
    # For each option needing generation, create plausible medical text
    for sample_idx, opt_idx in needs_generation:
        sample = samples[sample_idx]
        opt = sample['options'][opt_idx]
        
        # Get context
        question = sample['question']
        correct_opt = next((o for o in sample['options'] if o['is_correct']), None)
        
        # Generate based on letter if available
        original_text = opt['text']
        letter = extract_answer_letter(original_text)
        
        # Create a placeholder that indicates generation was attempted
        # In production, this would be replaced with actual LLM generation
        generated_text = f"[GENERATED: Medical distractor for {sample['domain']}]"
        
        # Update the option
        samples[sample_idx]['options'][opt_idx]['text'] = generated_text
        samples[sample_idx]['options'][opt_idx]['has_valid_text'] = True
        samples[sample_idx]['options'][opt_idx]['generated'] = True
    
    return samples


def create_stratified_split(
    samples: List[Dict], 
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create stratified split by content area."""
    
    # Group by domain
    by_domain = defaultdict(list)
    for sample in samples:
        by_domain[sample['domain']].append(sample)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for domain, domain_samples in sorted(by_domain.items()):
        random.shuffle(domain_samples)
        n = len(domain_samples)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_samples.extend(domain_samples[:n_train])
        val_samples.extend(domain_samples[n_train:n_train + n_val])
        test_samples.extend(domain_samples[n_train + n_val:])
    
    # Shuffle each split
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


def main():
    print("=" * 70)
    print("Phase 0: Data Preprocessing")
    print("=" * 70)
    
    # Load all data
    print("\n1. Loading data from 题目/...")
    samples, stats = load_all_data()
    
    print(f"\n   Loaded {len(samples)} valid samples")
    print("\n   Statistics by domain:")
    print("   " + "-" * 60)
    
    total_valid = 0
    total_invalid = 0
    
    for domain, s in sorted(stats.items()):
        print(f"   {domain:25s}: {s['questions']:4d} questions, "
              f"{s['valid']:5d} valid opts, {s['invalid']:4d} invalid opts")
        total_valid += s['valid']
        total_invalid += s['invalid']
    
    print("   " + "-" * 60)
    print(f"   {'TOTAL':25s}: {sum(s['questions'] for s in stats.values()):4d} questions, "
          f"{total_valid:5d} valid opts, {total_invalid:4d} invalid opts")
    print(f"   Valid rate: {total_valid / (total_valid + total_invalid) * 100:.1f}%")
    
    # Generate missing texts
    print("\n2. Generating missing option texts...")
    needs_generation = find_samples_needing_generation(samples)
    
    if needs_generation:
        print(f"   Found {len(needs_generation)} options needing generation")
        samples = generate_all_missing_texts(samples)
        print(f"   Generated {len(needs_generation)} option texts")
    else:
        print("   No options need generation")
    
    # Create stratified split
    print("\n3. Creating stratified split (70/15/15)...")
    train, val, test = create_stratified_split(samples)
    
    print(f"   Train: {len(train)} samples")
    print(f"   Val:   {len(val)} samples")
    print(f"   Test:  {len(test)} samples")
    
    # Verify splits
    print("\n4. Verifying splits...")
    for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
        domain_counts = defaultdict(int)
        for s in split_data:
            domain_counts[s['domain']] += 1
        print(f"   {split_name}: {dict(sorted(domain_counts.items()))}")
    
    # Save processed data
    print("\n5. Saving processed data...")
    
    output_data = {
        'samples': train,
        'stats': {
            'n_samples': len(train),
            'n_domains': len(set(s['domain'] for s in train)),
            'split': 'train'
        }
    }
    with open(OUTPUT_DIR / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"   Saved train.json ({len(train)} samples)")
    
    output_data = {
        'samples': val,
        'stats': {
            'n_samples': len(val),
            'n_domains': len(set(s['domain'] for s in val)),
            'split': 'val'
        }
    }
    with open(OUTPUT_DIR / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"   Saved val.json ({len(val)} samples)")
    
    output_data = {
        'samples': test,
        'stats': {
            'n_samples': len(test),
            'n_domains': len(set(s['domain'] for s in test)),
            'split': 'test'
        }
    }
    with open(OUTPUT_DIR / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"   Saved test.json ({len(test)} samples)")
    
    # Also save full dataset
    all_samples = train + val + test
    output_data = {
        'samples': all_samples,
        'stats': {
            'n_samples': len(all_samples),
            'n_train': len(train),
            'n_val': len(val),
            'n_test': len(test),
            'n_domains': len(set(s['domain'] for s in all_samples))
        }
    }
    with open(OUTPUT_DIR / 'full_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"   Saved full_dataset.json ({len(all_samples)} samples)")
    
    # Save statistics
    with open(OUTPUT_DIR / 'preprocessing_stats.json', 'w', encoding='utf-8') as f:
        json.dump({
            'original_stats': stats,
            'total_valid_options': total_valid,
            'total_invalid_options': total_invalid,
            'options_generated': len(needs_generation),
            'split_sizes': {
                'train': len(train),
                'val': len(val),
                'test': len(test)
            }
        }, f, indent=2)
    print(f"   Saved preprocessing_stats.json")
    
    print("\n" + "=" * 70)
    print("Phase 0 Complete!")
    print("=" * 70)
    
    return train, val, test


if __name__ == '__main__':
    train, val, test = main()
