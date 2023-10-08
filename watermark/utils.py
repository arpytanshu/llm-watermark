
import os
from pathlib import Path
from time import perf_counter
from typing import List, Dict, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark import WaterMark



def wmGenerate(model: AutoModelForCausalLM,
               tokenizer: AutoTokenizer,
               prompt: str,
               watermarker: WaterMark,
               temperature: float = 0.7,
               do_sample: bool = True,
               max_length: int = 100):
   
    device = model.device
    input_ids = tokenizer.encode(prompt,
                                 add_special_tokens=True, 
                                 return_tensors='pt').to(device)

    prompt_len = len(input_ids[0])
    batch_sz = input_ids.shape[0]

    for ix in range(max_length):
        with torch.no_grad():
            output = model(input_ids.to(model.device), 
                           output_hidden_states=True, 
                           use_cache=False)
            
        if watermarker is not None:
            logits = watermarker(output.logits, input_ids) # BxT
        else:
            logits = output.logits[:, -1, :] # BxT
        
        logits = logits / temperature
        probs = torch.softmax(logits, axis=1) # BxT
        
        if do_sample:
            out_token_id = torch.multinomial(probs, 1) # Bx1
        else:
            # greedy decoding
            out_token_id = torch.argmax(probs, axis=1).unsqueeze(1)

        input_ids = torch.cat([input_ids, out_token_id], dim=-1)
        
        if (batch_sz == 1) & \
            (out_token_id.item() == tokenizer.eos_token_id):   
            break
  
    prompt_ids = input_ids[:, :prompt_len]
    generation_ids = input_ids[:, prompt_len:]
    return prompt_ids, generation_ids


def prepare_llama_prompt(prompt:str) -> str:
    inst_bgn_tkn = "[INST]"
    inst_end_tkn = "[/INST]"
    return f"{inst_bgn_tkn} {prompt} {inst_end_tkn}"


def get_model(model_string:str, load_in_8bit: bool = True):
    tick = perf_counter()
    # torch_dtype=torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_string, 
                                                 load_in_8bit=load_in_8bit)
    tokenizer = AutoTokenizer.from_pretrained(model_string)
    elapsed = perf_counter() - tick
    n_params = model.num_parameters() // 1024 // 1024
    print(f"{elapsed=:.4f} s in loading model w/ {n_params}M parameters.")
    return model, tokenizer


def create_gif(stats: List[Dict],
               user_prompt: str, 
               tokenizer: AutoTokenizer, 
               dest_path: Union[str, Path]):
    def _format(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    tokens = [x['token_id'] for x in stats]
    z_stats = [x['z_stat'] for x in stats]
    results = [x['result'] for x in stats]
    p_vals = [x['p_val'] for x in stats]

    detection_ixs = np.where(np.array([x['result'] for x in stats]))[0]
    DETECTED = False

    max_lines = len(tokenizer.decode(tokens).split('\n')) + 1
    
    os.makedirs('/tmp/gifcache/', exist_ok=True)
    for ix in range(len(stats)):

        tokens_till_ix = tokens[:ix]
        z_stat_till_ix = np.array(z_stats[:ix])
        results_till_ix = np.array(results[:ix])

        z_stat = z_stats[ix]
        p_val = p_vals[ix]

        string = tokenizer.decode(tokens_till_ix)
        # print(string)
        string = [[y for y in x.split()] for x in string.split('\n')]

        fig, axs = plt.subplots(1, 2, figsize=(8,3), gridspec_kw={'width_ratios': [2, 1]})
        _format(axs[0])

        for line_ix, line in enumerate(string):
            line = ' '.join(line)                
            axs[0].text(x=0.05, y=1, s='Prompt: ' + user_prompt, c='b')
            axs[0].text(x=0.05, y=1-((line_ix+1)/max_lines), s=line)

        if z_stat > 4:
            DETECTED = True
            axs[0].text(0, 0-(1/max_lines), 'Z statistic: '+str(round(z_stat, 3)), c='r')
            axs[0].text(0, 0-(2/max_lines), 'p-val: '+str(round(p_val, 7)), c='r')
        else:
            axs[0].text(0, 0-(1/max_lines), 'Z statistic:'+str(round(z_stat, 3)), c='g')
            axs[0].text(0, 0-(2/max_lines), 'p-val: '+str(round(p_val, 7)), c='g')
        
        if DETECTED:
            axs[0].text(0, 0-(3/max_lines), f'WaterMark Detected @ {detection_ixs[0]}th tkn', c='r')
        
        ax = axs[1]
        ax.set_xticks(range(0, len(tokens), 15))
        ax.set_xlabel('token #')
        ax.set_ylabel('z statistic')
        ax.plot(z_stat_till_ix, color='k', alpha=0.5)

        if ix > 0:
            ax.scatter(np.where(results_till_ix)[0], z_stat_till_ix[np.where(results_till_ix)[0]], c='r', s=5)
            ax.scatter(np.where(~results_till_ix)[0], z_stat_till_ix[np.where(~results_till_ix)[0]], c='g', s=5)


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()

        plt.savefig(f'/tmp/gifcache/{ix}.png')
        plt.close()

    dest_path = Path(dest_path)
    os.system('ffmpeg -i /tmp/gifcache/%d.png -loglevel quiet -vf palettegen /tmp/gifcache/palette.png ')
    os.system(f'ffmpeg -y -s 800x300 -framerate 6 -loglevel quiet -i /tmp/gifcache/%d.png -i /tmp/gifcache/palette.png -lavfi paletteuse {dest_path}')
    os.system('rm /tmp/gifcache/*.png')
    