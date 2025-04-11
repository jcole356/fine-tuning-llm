from transformers import TrainingArguments
from peft import LoraConfig, TaskType


def get_lora_config():
    return LoraConfig(
        r=8,                     # LoRA attention dimension
        lora_alpha=32,          # Alpha parameter for LoRA scaling
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


# output_dir is where the model checkpoints will be saved during training
def get_training_args():
    return TrainingArguments(
        output_dir="lora_hr_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        warmup_steps=100,
        weight_decay=0.01,
    )
