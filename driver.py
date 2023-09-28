#%%

import torch
from utils import get_model, prepare_llama_prompt
from copy import deepcopy
from watermark import (
    WaterMark, 
    WaterMarkConfig, 
    wm_generate
)

torch.set_default_device('cuda')
# MODEL_STRING = 'cerebras/Cerebras-GPT-111M'
# MODEL_STRING = "meta-llama/Llama-2-7b-chat-hf"
MODEL_STRING = "meta-llama/Llama-2-13b-chat-hf"
model, tokenizer = get_model(MODEL_STRING)


#%%

wm_cfg = WaterMarkConfig(vocab_size=tokenizer.vocab_size)
wm_cfg.soft_mode = True
wm_cfg.hardness = 1.5

watermarker = WaterMark(wm_cfg)
prompt = 'write a 4 liner poetry about tensors.'
prompt = prepare_llama_prompt(prompt)

generation_config = {'max_length':  100, 
                     'temperature': 0.7, 
                     'do_sample':   False, 
                     'model':       model,
                     'tokenizer':   tokenizer,
                     'prompt':      prompt
                     }

# get watermarked generations
prompt_ids, wm_output_ids = wm_generate(**generation_config, watermarker=watermarker)

# get non-watermarked generations
prompt_ids, output_ids = wm_generate(**generation_config, watermarker=None, )

prompt = tokenizer.decode(prompt_ids.squeeze(0), skip_special_tokens=True)
non_wm_gens = tokenizer.decode(output_ids.squeeze(0), skip_special_tokens=True)
wm_gens = tokenizer.decode(wm_output_ids.squeeze(0), skip_special_tokens=True)

print(prompt)
print('='*16)
print()
print('WaterMarked Generations:')
print('='*16)
print(wm_gens)
print()
print('non-WaterMarked Generations:')
print('='*16)
print(non_wm_gens)
print()


# %%
