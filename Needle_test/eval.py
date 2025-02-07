import yaml
import os
import json
import re
from collections import defaultdict

def clean_text(text):
    """Lowercase text, remove punctuation, and normalize spacing."""
    text = text.lower()  # Ignore case
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

def evaluate_predictions(pred_base_dir, reference, k_values):
    """
    Evaluate predictions for each K value directory.
    
    Args:
        pred_base_dir (str): Base directory containing K-specific folders
        reference (str): Reference text to search for in predictions
        k_values (list): List of K values to evaluate
        
    Returns:
        dict: Results organized by K value
    """
    # Clean reference text once since it's used multiple times
    cleaned_reference = clean_text(reference)
    
    # Store results for all K values
    all_results = {}
    
    # Process each K directory
    for k in k_values:
        k_dir = f'{pred_base_dir}_K{k}'
        if not os.path.exists(k_dir):
            print(f"Warning: Directory for K={k} not found at {k_dir}")
            continue
            
        k_results = {}
        
        #process each prediction file in the K directory
        for filename in os.listdir(k_dir):
            if not filename.endswith('.txt'):
                continue
                
            with open(os.path.join(k_dir, filename), 'r') as f:
                prediction = f.read().strip()
            
            cleaned_prediction = clean_text(prediction)
            score = 10 if cleaned_reference in cleaned_prediction else 0
            
            k_results[filename.replace('.txt', '')] = {
                'prediction': prediction,
                'score': score
            }
        
        all_results[f'K{k}'] = k_results
    
    return all_results

def compute_summary_statistics(results):
    summary_stats = {}
    
    for k_value, k_results in results.items():
        total_predictions = len(k_results)
        successful_predictions = sum(1 for pred in k_results.values() if pred['score'] == 10)
        
        summary_stats[k_value] = {
            'total_predictions': total_predictions,
            'successful_predictions': successful_predictions,
            'success_rate': successful_predictions / total_predictions if total_predictions > 0 else 0
        }
    
    return summary_stats

if __name__ == '__main__':
    # Load configuration
    with open('LongAlign/Needle_test/config-eval.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    pred_base_dir = config['pred_dir']  # Base directory path without K suffix
    save_dir = config['save_dir']
    reference = config['prompt']['needle']
    
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get K values from directory names
    k_pattern = re.compile(r'_K(\d+)$')
    k_values = []
    
    # Search for K directories in the parent directory of pred_base_dir
    parent_dir = os.path.dirname(pred_base_dir)
    base_name = os.path.basename(pred_base_dir)
    
    for dirname in os.listdir(parent_dir):
        if dirname.startswith(base_name):
            match = k_pattern.search(dirname)
            if match:
                k_values.append(int(match.group(1)))
    
    k_values.sort()  # Sort K values numerically
    
    if not k_values:
        print("Error: No K directories found!")
        exit(1)
    
    print(f"Found K values: {k_values}")
    
    # Evaluate predictions for all K values
    results = evaluate_predictions(pred_base_dir, reference, k_values)
    
    # Compute summary statistics
    summary_stats = compute_summary_statistics(results)
    
    # Save detailed results
    with open(os.path.join(save_dir, 'binary_eval_all_k.json'), 'w') as f:
        json.dump({
            'detailed_results': results,
            'summary_statistics': summary_stats
        }, f, indent=4)
    
    # Print summary statistics
    print("\nEvaluation Summary:")
    for k_value, stats in summary_stats.items():
        print(f"\n{k_value}:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Successful predictions: {stats['successful_predictions']}")
        print(f"  Success rate: {stats['success_rate']*100:.2f}%")
