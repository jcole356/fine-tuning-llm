from huggingface_hub import HfApi

api = HfApi()
# Should print your username if logged in
print(api.whoami())
