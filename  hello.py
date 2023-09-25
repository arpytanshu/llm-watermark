#%%
import torch
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device('cuda')

tick = perf_counter()
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")


elapsed = perf_counter() - tick
print(f"{elapsed=:.4f}")

#%%

inputs = tokenizer('''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)


outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)



