
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def prepare_llama_prompt(prompt:str) -> str:
    inst_bgn_tkn = "[INST]"
    inst_end_tkn = "[/INST]"
    return f"{inst_bgn_tkn} {prompt} {inst_end_tkn}"

def get_model(MODEL_STRING):
   tick = perf_counter()
   # torch_dtype=torch.bfloat16
   model = AutoModelForCausalLM.from_pretrained(MODEL_STRING, load_in_8bit=True)
   tokenizer = AutoTokenizer.from_pretrained(MODEL_STRING)
   elapsed = perf_counter() - tick
   n_params = model.num_parameters() // 1024 // 1024
   print(f"{elapsed=:.4f} s in loading model w/ {n_params}M parameters.")
   return model, tokenizer