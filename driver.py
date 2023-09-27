#%%
import torch
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark import WaterMarking, WaterMarkingConfig

torch.set_default_device('cuda')

tick = perf_counter()
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
elapsed = perf_counter() - tick
print(f"{elapsed=:.4f}")


wm_cfg = WaterMarkingConfig(vocab_size=tokenizer.vocab_size)

watermarker = WaterMarking(wm_cfg)


#%%


input_ids = tokenizer('''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)['input_ids']

output = model(input_ids, output_hidden_state=True)


