def augment_dataset(dataset):
    """Augment dataset with variations of existing examples"""
    augmented_data = []


    for example in dataset:
        # Original example
        augmented_data.append(example)


        # Create variation with different phrasing
        if example['topic'] == 'leave_policy':
            augmented_data.append({
                'instruction': f"Could you explain {example['instruction'].lower()}",
                'response': example['response'],
                'topic': example['topic']
            })


    return augmented_data
