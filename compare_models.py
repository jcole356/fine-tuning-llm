import json
from evaluate_model import ModelEvaluator
from typing import Dict


def compare_models():
    print('Comparing models...')
    
    # Load both base and fine-tuned models
    base_evaluator = ModelEvaluator('gpt2')  # Base model
    tuned_evaluator = ModelEvaluator('final_lora_model')


    # Run comparisons
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)


    base_results = base_evaluator.evaluate_responses(test_cases['test_cases'])
    tuned_results = tuned_evaluator.evaluate_responses(test_cases['test_cases'])


    # Generate comparison report
    generate_comparison_report(base_results, tuned_results)


def generate_comparison_report(base_results: Dict, tuned_results: Dict):
    """
    Generate a comparison report between the base model and the fine-tuned model.
    """
    print("Comparison Report:")

    # Calculate average ROUGE and BERT scores for both models
    def calculate_average_scores(results, metric_key):
        return {
            key: sum(score[key] for score in results[metric_key]) / len(results[metric_key])
            for key in results[metric_key][0]
        }

    base_avg_rouge = calculate_average_scores(base_results, 'rouge_scores')
    tuned_avg_rouge = calculate_average_scores(tuned_results, 'rouge_scores')

    base_avg_bert = calculate_average_scores(base_results, 'bert_scores')
    tuned_avg_bert = calculate_average_scores(tuned_results, 'bert_scores')

    # Print ROUGE score comparison
    print("\nROUGE Score Comparison:")
    for key in base_avg_rouge:
        print(f"  {key}:")
        print(f"    Base Model: {base_avg_rouge[key]:.4f}")
        print(f"    Fine-Tuned Model: {tuned_avg_rouge[key]:.4f}")
        print(f"    Improvement: {tuned_avg_rouge[key] - base_avg_rouge[key]:.4f}")

    # Print BERT score comparison
    print("\nBERT Score Comparison:")
    for key in base_avg_bert:
        print(f"  {key}:")
        print(f"    Base Model: {base_avg_bert[key]:.4f}")
        print(f"    Fine-Tuned Model: {tuned_avg_bert[key]:.4f}")
        print(f"    Improvement: {tuned_avg_bert[key] - base_avg_bert[key]:.4f}")

    # Print example response comparisons
    print("\nExample Response Comparisons:")
    for i, (base_example, tuned_example) in enumerate(zip(base_results['response_examples'], tuned_results['response_examples'])):
        if i >= 3:  # Limit to 3 examples
            break
        print(f"Example {i + 1}:")
        print(f"  Instruction: {base_example['instruction']}")
        print(f"  Base Model Response: {base_example['generated']}")
        print(f"  Fine-Tuned Model Response: {tuned_example['generated']}")
        print(f"  Expected Response: {base_example['expected']}")
        print("-" * 50)

if __name__ == "__main__":
    compare_models()
