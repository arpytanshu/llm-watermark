#%%

import torch
from utils import get_model, prepare_llama_prompt, plot_stats
from copy import deepcopy
from watermark import (
    WaterMark, 
    WaterMarkConfig, 
    WaterMarkDetector,
    wmGenerate
)

MODEL_STRING = 'cerebras/Cerebras-GPT-111M'
# MODEL_STRING = "meta-llama/Llama-2-7b-chat-hf"
# MODEL_STRING = "meta-llama/Llama-2-13b-chat-hf"
DEVICE = torch.device('cuda:0')

model, tokenizer = get_model(MODEL_STRING, load_in_8bit=False)
model.to(DEVICE)

#%%

wm_cfg = WaterMarkConfig(vocab_size=tokenizer.vocab_size)
wm_cfg.soft_mode = True # hard_mode
wm_cfg.hardness = 2.0
wm_cfg.detection_device = model.device

watermarker = WaterMark(wm_cfg)
wm_detector = WaterMarkDetector(wm_cfg)


prompt = 'write a 3 liner poetry about tensors.'
prompt = prepare_llama_prompt(prompt)

generation_config = {
    'max_length':  100, 
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

stats1 = wm_detector.detect(prompt, wm_gens, tokenizer)
stats2 = wm_detector.detect(prompt, non_wm_gens, tokenizer)


#%%
import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(1,1, figsize=(10,5))
plot_stats(stats1, ax, 'watermarked generations')
plot_stats(stats2, ax, 'non watermarked generations')


# %%


import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot()
fig.subplots_adjust(top=0.85)

# Set titles for the figure and the subplot respectively
fig.suptitle('suptitle', fontsize=14, fontweight='bold')
ax.set_title('axes title')

ax.set_xticks([])
ax.set_yticks([])


# ax.text(.3, .8, 'boxed italics text in data coords', style='italic',
#         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

ax.text(0, 1, 'Prompt Text.',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='green', fontsize=15)

ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)


plt.show()


# %%
