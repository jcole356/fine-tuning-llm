from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "tiiuae/falcon-7b"
model_name = "gpt2"
# model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✓ Model access verified")
