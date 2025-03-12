from datasets import load_dataset, ClassLabel
import pandas as pd
from sklearn.model_selection import train_test_split


def validate_data_entry(entry):
    """Validate each data entry for required fields and format"""
    required_fields = ['instruction', 'response']
    return all(field in entry and len(entry[field].strip()) > 0 
              for field in required_fields)


def analyze_dataset_distribution(dataset):
    """Analyze topic distribution in dataset"""
    if 'topic' in dataset['train'].features:
        topic_dist = dataset['train'].to_pandas()['topic'].value_counts()
        print("Topic Distribution:")
        print(topic_dist)


def prepare_dataset():
    # Load the dataset
    try:
        dataset = load_dataset('json', data_files='training_data.jsonl')
    except FileNotFoundError:
        print("The file 'training_data.jsonl' was not found.")
        return None
    
    print('DATASET')
    print(dataset)

    # Validate entries
    valid_entries = dataset['train'].filter(validate_data_entry)

    print('VALID ENTRIES')
    print(valid_entries)

    valid_entries = dataset['train'].filter(validate_data_entry)

    # Convert 'topic' column to ClassLabel if it exists
    if 'topic' in valid_entries.features:
        topic_classes = pd.Series(valid_entries['topic']).unique().tolist()
        print('TOPIC CLASSES')
        print(topic_classes)
        valid_entries = valid_entries.cast_column('topic', ClassLabel(names=topic_classes))


    # Split dataset
    try:
      train_test = valid_entries.train_test_split(
          test_size=0.1, 
          seed=42,
          stratify_by_column='topic' if 'topic' in valid_entries.features else None
      )
    except ValueError as e:
        print(f"Error during train_test_split: {e}")
        return None

    print('TRAIN TEST')
    print(train_test)


    # Analyze distribution
    analyze_dataset_distribution(train_test)


    # Save processed dataset
    train_test.save_to_disk('hr_qa_dataset')


    return train_test

if __name__ == "__main__":
    dataset = prepare_dataset()
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['test'])}")
