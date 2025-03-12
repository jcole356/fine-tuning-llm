from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tiiuae/falcon-7b"
# model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✓ Model access verified")
