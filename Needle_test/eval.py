import json
import os
import re
from pathlib import Path

import yaml


CONF_FILE = "config.yaml"


def clean_text(text):
    """Lowercase text, remove punctuation, and normalize spacing."""
    text = text.lower()  # Ignore case
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text


def evaluate_predictions(pred_base_dir, reference, k_values):
    """Evaluate predictions for each K value directory.

    This assumes that predictions are sorted
    in directories of the format "{pred_base_dir}_K{k_values}".


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
        k_dir = f"{pred_base_dir}/K{k}"
        if not os.path.exists(k_dir):
            print(f"Warning: Directory for K={k} not found at {k_dir}")
            continue

        k_results = {}

        # process each prediction file in the K directory
        for filename in os.listdir(k_dir):
            if not filename.endswith(".txt"):
                continue

            with open(os.path.join(k_dir, filename), "r") as f:
                prediction = f.read().strip()

            cleaned_prediction = clean_text(prediction)

            score = 0
            if "best thing to do in san francisco" in cleaned_prediction:
                score += 1
            if "eat a sandwich" in cleaned_prediction:
                score += 1
            if "dolores park" in cleaned_prediction:
                score += 0.5
                if "sit in dolores park" in cleaned_prediction:
                    score += 0.5
            if "on a sunny day" in cleaned_prediction:
                score += 1
            if cleaned_reference in cleaned_prediction:
                score = 5

            # if score == 0:
            #     print(f"reference: {cleaned_reference}")
            #     print(f"prediction: {half_prediction}")

            k_results[filename.replace(".txt", "")] = {
                "prediction": prediction,
                "score": score,
            }

        all_results[f"K{k}"] = k_results

    return all_results


def compute_summary_statistics(results):
    summary_stats = {}

    for k_value, prompt_results in results.items():
        total_predictions = len(prompt_results)
        successful_predictions = sum(1 for pred in prompt_results.values() if pred["score"] == 5)
        average_points = (
            sum(pred["score"] for pred in prompt_results.values()) / total_predictions / 5
        )

        summary_stats[k_value] = {
            "total_predictions": total_predictions,
            "successful_predictions": successful_predictions,
            "average_points": average_points,
        }

    return summary_stats


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).resolve().parent / CONF_FILE
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    parent_dir = Path(config["parent_dir"])
    pred_base_dir = parent_dir / config["pred"]["save_dir"]
    save_dir = parent_dir / config["eval"]["save_dir"]
    reference = config["prompt"]["needle"]

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get K values from directory names
    k_pattern = re.compile(r"K(\d+)$")
    k_values = []

    for dirname in os.listdir(pred_base_dir):
        if dirname.startswith("K"):
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
    for key in results:
        with open(os.path.join(save_dir, f"binary_eval_K{key[1:]}.json"), "w") as f:
            json.dump(
                {
                    "detailed_results": results[key],
                    "summary_statistics": summary_stats[key],
                },
                f,
                indent=4,
            )

    # Print summary statistics
    print("\nEvaluation Summary:")
    for k_value, stats in summary_stats.items():
        print(f"\n{k_value}:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Successful predictions: {stats['successful_predictions']}")
        print(f"  Success rate: {stats['average_points'] * 100:.2f}%")
