import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from peft import get_peft_model
from datasets import load_from_disk
from lora_config import get_lora_config, get_training_args

def prepare_model():
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b",
        torch_dtype=torch.float16,
        device_map="auto"
    )


    # Apply LoRA config
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)


    return model


def train():
    # Load model and tokenizer
    model = prepare_model()
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")


    # Load dataset
    dataset = load_from_disk('hr_qa_dataset')


    # Training arguments
    training_args = get_training_args()


    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )


    # Train and save the final model
    trainer.train()
    trainer.save_model("final_lora_model")


if __name__ == "__main__":
    train()
