import json

def reformat_hr_qa_data(input_file: str, output_file: str):
    """
    Reformat the JSON data from hr_qa_data.jsonl to match the format in training_data.jsonl.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to save the reformatted JSONL file.
    """
    with open(input_file, 'r') as infile:
        data = json.load(infile)  # Load the JSON data

    reformatted_data = []
    for item in data:
        # Map fields to the new format
        reformatted_item = {
            "question": item["prompt"].split('. ', 1)[-1],  # Remove the number and period from the beginning of "prompt"
            "answer": item["response"],  # Keep "response" unchanged
            "category": item["source_document"].replace('-', '_').replace('.docx', '').lower()  # Remove ".docx" and convert to lowercase
        }
        reformatted_data.append(reformatted_item)

    # Save the reformatted data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(reformatted_data, outfile, indent=2)

# Example usage
# reformat_hr_qa_data('hr_qa_data.jsonl', 'training_data.jsonl')

if __name__ == "__main__":
    reformat_hr_qa_data('hr_qa_data.jsonl', 'test_training_data.jsonl')
