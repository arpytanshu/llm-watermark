#%%
import torch
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark import WaterMarking, WaterMarkingConfig
from watermark import WaterMarking, WaterMarkingConfig
from copy import deepcopy

torch.set_default_device('cuda')

MODEL_STRING = 'cerebras/Cerebras-GPT-111M'

tick = perf_counter()
model = AutoModelForCausalLM.from_pretrained(MODEL_STRING)
tokenizer = AutoTokenizer.from_pretrained(MODEL_STRING)
elapsed = perf_counter() - tick
n_params = model.num_parameters() // 1024 // 1024
print(f"{elapsed=:.4f} s in loading model w/ {n_params}M parameters.")


wm_cfg = WaterMarkingConfig(vocab_size=tokenizer.vocab_size)
watermarker = WaterMarking(wm_cfg)


#%%

input_ids = tokenizer("write a 4 line poetry about tensors.", 
                      return_tensors="pt")['input_ids']
output = model(input_ids, output_hidden_states=True)
wm_logits = watermarker(output.logits, input_ids)


# %%

def wm_generate(model: AutoModelForCausalLM,
                prompt: str,
                temperature: float = 0.7,
                max_length: int = 100,
): 
    device = model.device()
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt').to(device)
    prompt_len = len(input_ids[0])

    for ix in range(max_length):
        with torch.no_grad():

            out = model(torch.tensor(input_ids).to(torch.int).unsqueeze(0).to(model.device),
                        output_hidden_states=True,
                        use_cache=False)
            
            wm_logits = watermarker(output.logits, input_ids)
            probs = torch.softmax(wm_logits.squeeze()/temperature, axis=0)
            out_token_id = torch.multinomial(probs, 1)[0].item()
            
        if out_token_id == tokenizer.eos_token_id:
            break;e
        else:
            input_ids.append(out_token_id)

