import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import evaluate
import json
from typing import List, Dict

# TODO: This is the next task
class ModelEvaluator:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")


        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Pad the base model to match vocab size of the fine-tuned model
        self.tokenizer.add_special_tokens({'pad_token': '</s>'})
        base_model.resize_token_embeddings(len(self.tokenizer))


        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            base_model,
            model_path,
            device_map="auto"
        )


        # Evaluation metrics
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')


    def generate_response(self, prompt: str) -> str:
        # Tokenize the input and move it to the correct device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Ensure the model is on the correct device
        self.model.to(self.device)

        # Generate the response
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.7,
            num_beams=4,
            no_repeat_ngram_size=2
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def evaluate_responses(self, test_cases: List[Dict]) -> Dict:
        results = {
            'accuracy': 0,
            'rouge_scores': [],
            'bert_scores': [],
            'response_examples': []
        }


        for case in test_cases:
            response = self.generate_response(case['question'])


            # Calculate metrics
            rouge_score = self.rouge.compute(
                predictions=[response],
                references=[case['answer']]
            )


            bert_score = self.bertscore.compute(
                predictions=[response],
                references=[case['answer']],
                lang='en'
            )

            results['rouge_scores'].append(rouge_score)
            results['bert_scores'].append(bert_score)
            results['response_examples'].append({
                'question': case['question'],
                'answer': case['answer'],
                'generated': response
            })


        return results


def print_evaluation_summary(results: Dict):
    # Print detailed scores
    print("ROUGE Scores:", results['rouge_scores'])
    print("BERT Scores:", results['bert_scores'])
    
    """
    Print a summary of the evaluation results, including average scores and example responses.
    """
    # Calculate average ROUGE and BERT scores
    # avg_rouge = {
    #     key: sum(score[key] for score in results['rouge_scores']) / len(results['rouge_scores'])
    #     for key in results['rouge_scores'][0]
    # }
    # avg_bert = {
    #     key: sum(score[key] for score in results['bert_scores']) / len(results['bert_scores'])
    #     for key in results['bert_scores'][0]
    # }

    # # Print average scores
    # print("Evaluation Summary:")
    # print("Average ROUGE Scores:")
    # for key, value in avg_rouge.items():
    #     print(f"  {key}: {value:.4f}")

    # print("\nAverage BERT Scores:")
    # for key, value in avg_bert.items():
    #     print(f"  {key}: {value:.4f}")

    # # Print detailed scores
    # print("ROUGE Scores:", results['rouge_scores'])
    # print("BERT Scores:", results['bert_scores'])

    # # Print a few example responses
    # print("\nExample Responses:")
    # for example in results['response_examples'][:3]:  # Print the first 3 examples
    #     print(f"Instruction: {example['instruction']}")
    #     print(f"Expected: {example['expected']}")
    #     print(f"Generated: {example['generated']}")
    #     print("-" * 50)


def main():
    # Load test cases
    with open('test_cases.json', 'r') as f:
        test_cases = json.load(f)

    # Debugging: Check the structure of test_cases
    print(type(test_cases))  # Should be <class 'list'>
    # print(test_cases[:2])    # Print the first two test cases for verification

    # Initialize evaluator
    # evaluator = ModelEvaluator('final_lora_model') # Trained local model
    evaluator = ModelEvaluator('fisherscats/fine-tuned-hr-qa-model') # Trained remote model

    # Run evaluation
    results = evaluator.evaluate_responses(test_cases['test_cases'])

    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print_evaluation_summary(results)


if __name__ == "__main__":
    main()
