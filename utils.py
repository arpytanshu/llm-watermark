
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_llama_prompt(prompt:str) -> str:
    inst_bgn_tkn = "[INST]"
    inst_end_tkn = "[/INST]"
    return f"{inst_bgn_tkn} {prompt} {inst_end_tkn}"

def get_model(MODEL_STRING, load_in_8bit=True):
   tick = perf_counter()
   # torch_dtype=torch.bfloat16
   model = AutoModelForCausalLM.from_pretrained(MODEL_STRING, load_in_8bit=load_in_8bit)
   tokenizer = AutoTokenizer.from_pretrained(MODEL_STRING)
   elapsed = perf_counter() - tick
   n_params = model.num_parameters() // 1024 // 1024
   print(f"{elapsed=:.4f} s in loading model w/ {n_params}M parameters.")
   return model, tokenizer



def plot_stats(stats, ax=None, title=''):
    z_stats = np.array([x['z_stat'] for x in stats])
    result = np.array([x['result'] for x in stats])
    if ax is None:
        plt.figure(figsize=(10, 5))
        plt.scatter(np.where(result)[0], z_stats[np.where(result)[0]], c='r', s=5)
        plt.scatter(np.where(~result)[0], z_stats[np.where(~result)[0]], c='g', s=5)
        plt.title(title)
        plt.show()
    else:
        ax.scatter(np.where(result)[0], z_stats[np.where(result)[0]], c='r', s=5)
        ax.scatter(np.where(~result)[0], z_stats[np.where(~result)[0]], c='g', s=5)
    