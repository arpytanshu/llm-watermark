
import fire
import torch
from pathlib import Path
from typing import Union
import numpy as np

from watermark import (
    WaterMark, 
    WaterMarkConfig, 
    WaterMarkDetector,
    wmGenerate,
    get_model,
    prepare_llama_prompt,
    create_gif
)

def run(
    user_prompt: str,
    model_string: str = "meta-llama/Llama-2-7b-chat-hf",
    device: torch.device = torch.device('cuda'),
    watermark: bool = True,         # [True, False]
    watermark_mode: str = 'soft',   # ['hard', 'soft']
    hardness: int = 4.0,
    max_length: int = 300,
    do_sample: bool = True,
    temperature: float = 0.1,
    save_gif: bool = True,
    gif_dest_path: Union[Path, str] = Path('/tmp')
):
    """
    Run watermarking with the given user prompt.

    Parameters:
    -----------
    user_prompt : str
        The prompt for the LLM.
    model_string : str (optional)
        The hf-model to use. Defaults to "meta-llama/Llama-2-13b-chat-hf".
    device : torch.device (optional)
        The device to use. Defaults to CUDA.
    watermark : bool (optional)
        Whether to add a watermark to the output. Defaults to True.
    watermark_mode : str (optional)
        The mode for the watermark. Defaults to "soft".
    hardness : int (optional)
        The hardness of the watermark. Defaults to 4.0.
    max_length : int (optional)
        The maximum length of the model generations. Defaults to 250.
    do_sample : bool (optional)
        Whether to enable sampling. Defaults to True. Use False for greedy sampling.
    temperature : float (optional)
        The temperature for the generations. Defaults to 0.7.
    create_gif : bool (optional)
        Whether to create a GIF of the output. Defaults to True.
    gif_dest_path : Union[Path, str] (optional)
        The path to save the GIF to. Defaults to /tmp.

    Returns:
    -------
    output : str
        The output of the chatbot.
    """
    # initializations
    model, tokenizer = get_model(model_string, load_in_8bit=True)
    prompt = prepare_llama_prompt(user_prompt)

    # create watermarker object
    wm_cfg = WaterMarkConfig(vocab_size=tokenizer.vocab_size, device=model.device)
    if watermark:
        print('WaterMarking has been enabled.')
        if watermark_mode == 'soft':
            wm_cfg.soft_mode = True
            wm_cfg.hardness = hardness
            print(f'WaterMarking Mode: soft w/ {hardness=}')
        elif watermark_mode == 'hard':
            wm_cfg.soft_mode = False
            print('WaterMarking Mode: hard')
        else:
            raise Exception(f"Expected watermark_mode to be one of ['hard', 'soft'] but found {watermark_mode=}.")
        watermarker = WaterMark(wm_cfg)
    else:
        watermarker = None
        print('WaterMarking has been disabled.')
    
    # get watermarked generations
    prompt_ids, output_ids = wmGenerate(
        model=model, 
        tokenizer=tokenizer, 
        prompt=prompt, 
        watermarker=watermarker,
        max_length=max_length, 
        temperature=temperature, 
        do_sample=do_sample)

    prompt = tokenizer.decode(prompt_ids.squeeze(0), skip_special_tokens=True)
    generations = tokenizer.decode(output_ids.squeeze(0), skip_special_tokens=True)

    # run detections
    wm_detector = WaterMarkDetector(wm_cfg)
    detection_stats = wm_detector.detect(prompt, generations, tokenizer)

    # output stuffs
    print('\n\n'+'-'*8)
    print('PROMPT::', user_prompt)
    print('GENERATIONS::')
    print(generations)
    print('---- end of generation ----')

    detection_ixs = np.where(np.array([x['result'] for x in detection_stats]))[0]
    if len(detection_ixs) > 0:
        print(f'Watermark was detected at the {detection_ixs[0]}th token.')
    else:
        print(f'Watermark was not detected.')
    print('\n\n'+'-'*8)


    if save_gif:
        file_name = f'wm:{watermark}'
        if watermark: 
            file_name += f'_mode:{watermark_mode}'
            if watermark_mode == 'soft':
                file_name += f'_hardness:{hardness}'
        file_name += '.gif'
        gif_dest_path = Path(gif_dest_path) / file_name

        create_gif(stats=detection_stats, 
                   user_prompt=user_prompt,
                   tokenizer=tokenizer, 
                   dest_path=gif_dest_path,
                   title=file_name[:-4])
        print(f'GIF saved at {gif_dest_path=}')


if __name__ == '__main__':
  fire.Fire(run)

