import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from peft import get_peft_model, PeftConfig
from datasets import load_from_disk
from lora_config import get_lora_config, get_training_args


def prepare_model():
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        # "tiiuae/falcon-1b",
        "gpt2",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Apply LoRA config
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    return model


def preprocess_function(examples, tokenizer):
    # Tokenize the inputs and labels
    inputs = tokenizer(examples['instruction'], padding="max_length", truncation=True)
    labels = tokenizer(examples['response'], padding="max_length", truncation=True)

    # Add labels to inputs
    inputs["labels"] = labels["input_ids"]
    return inputs


def train():
    # Load model and tokenizer
    model = prepare_model()
    # tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-1b")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_from_disk('hr_qa_dataset')

    # Preprocess dataset
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = get_training_args()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer
    )

    # Train and save
    trainer.train()
    trainer.save_model("final_lora_model")


if __name__ == "__main__":
    train()
