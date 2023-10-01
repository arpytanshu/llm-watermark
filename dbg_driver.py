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
# MODEL_STRING = "meta-llama/Llama-2-7b-chat-hf"
# model, tokenizer = get_model(MODEL_STRING)


class Model:
    def __init__(self, vocab):
        self.vocab = vocab
        self.device = torch.device('cpu')
    
    def __call__(self, inputs):
        b, t = inputs.shape
        return torch.rand(b, t, self.vocab)

class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
    
    def encode(self, string, add_special_tokens, return_tensors):
        return torch.randint(0, 32, (1,10))

model = Model(32)
tokenizer = Tokenizer(32)



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

#%%
temperature = 0.7
do_sample = False
max_length = 100



device = model.device
input_ids = tokenizer.encode(prompt, 
                            add_special_tokens=True, 
                            return_tensors='pt').to(device)

prompt_len = len(input_ids[0])
batch_sz = input_ids.shape[0]

input_ids.shape[0]
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
        break;

prompt_ids = input_ids[:, :prompt_len]
generation_ids = input_ids[:, prompt_len:]













#%%
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
