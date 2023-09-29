#%%

import torch
from utils import get_model, prepare_llama_prompt
from copy import deepcopy
from watermark import (
    WaterMark, 
    WaterMarkConfig, 
    WaterMarkDetector,
    wmGenerate
)

# torch.('cuda')
# MODEL_STRING = 'cerebras/Cerebras-GPT-111M'
# MODEL_STRING = "meta-llama/Llama-2-7b-chat-hf"
MODEL_STRING = "meta-llama/Llama-2-13b-chat-hf"
model, tokenizer = get_model(MODEL_STRING)

#%%

wm_cfg = WaterMarkConfig(vocab_size=tokenizer.vocab_size)
wm_cfg.soft_mode = False
wm_cfg.hardness = 2.5

watermarker = WaterMark(wm_cfg)
wm_detector = WaterMarkDetector(wm_cfg)

prompt = 'write a 8 liner poetry about tensors.'
prompt = prepare_llama_prompt(prompt)

generation_config = {
    'max_length':  200, 
    'temperature': 0.7, 
    'do_sample': False}

# get watermarked generations
prompt_ids, wm_output_ids = wmGenerate(
        model=model, 
        tokenizer=tokenizer, 
        prompt=prompt, 
        watermarker=watermarker, 
        **generation_config)

# get non-watermarked generations
prompt_ids, output_ids = wmGenerate(
        model=model, 
        tokenizer=tokenizer, 
        prompt=prompt, 
        watermarker=None,
        **generation_config)

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


stats1 = wm_detector.detect(prompt, wm_gens, tokenizer)
stats1


#%%
stats2 = wm_detector.detect(prompt, non_wm_gens, tokenizer)
stats2
 # %%
