
import torch
from typing import Callable, Union
from transformers import AutoTokenizer
from math import sqrt
from scipy.stats import norm


class WaterMarkConfig:
   def __init__(self, 
                vocab_size: int, 
                gamma: float = 0.5, 
                hardness: float = 4.0,
                hash_fn: Callable = hash,
                soft_mode: bool = True,
                device: Union[str, torch.device] = None):
      self.vocab_size = vocab_size
      self.gamma = gamma
      self.hardness = hardness # delta
      self.soft_mode = soft_mode
      self.hash_fn = hash_fn

      # G and R list sizes
      self.G_list_size = int(self.vocab_size * self.gamma)
      self.R_list_size = self.vocab_size - self.G_list_size

      # detection config
      self.threshold = 4
      self.detection_device = device

   def __repr__(self):
      return \
         f"""WaterMarkingConfig(vocab_size={self.vocab_size},
         mode={'soft' if self.soft_mode else 'hard'},
         gList={self.G_list_size},
         rList={self.R_list_size}"""


class WaterMark:
   def __init__(self, cfg: WaterMarkConfig):
      self.cfg = cfg
   
   def _hash_input(self, input_id):
      '''
      input_id: expected_shape is Bx1
      step 3 of Algorithm 2, which creates R list & G list.
      '''
      device = input_id.device
      g = torch.Generator(device)
      bch_sz = input_id.shape[0]
      G_list = torch.empty((bch_sz, 
                            self.cfg.G_list_size), 
                            dtype=torch.long, 
                            device=device) #
      R_list = torch.empty((bch_sz, 
                            self.cfg.R_list_size), 
                            dtype=torch.long, 
                            device=device) #
      
      for ix, id in enumerate(input_id):
         g.manual_seed(self.cfg.hash_fn(id.item()))
         rand_perm = torch.randperm(n=self.cfg.vocab_size, 
                                    generator=g, 
                                    device=device) #
         G_list[ix] = rand_perm[:self.cfg.G_list_size]
         R_list[ix] = rand_perm[self.cfg.G_list_size:]
      
      return {'R_list': R_list, 'G_list': G_list}
   
   def _hard_mode(self, logits, lists):
      logits.scatter_(index=lists['R_list'], dim=1, value=0)
   
   def _soft_mode(self, logits, lists):
      index = lists['G_list']
      src = torch.ones(index.shape, 
                       dtype=logits.dtype, 
                       device=logits.device) * self.cfg.hardness #
      logits.scatter_reduce_(dim=1, index=index, src=src, reduce='sum')

   def __call__(self, 
                logits: torch.FloatTensor,
                input_ids: torch.LongTensor):
            
      assert logits.dim() == 3, \
         f"Expected logits to have dimension 3, \
            but got dimension {logits.dim()}"
      
      assert input_ids.dim() == 2, \
         f"Expected input_ids to have dimension 2, \
            but got dimension {input_ids.dim()}"
      
      assert input_ids.device == logits.device, \
         f"Expected logits & input_ids to be on same device, \
            but found {input_ids.device.type} & {logits.device.type}"

      last_input_ids = input_ids[:, -1].view(-1, 1) 
      lists = self._hash_input(last_input_ids)

      logits = logits[:, -1, :] # 
      
      if self.cfg.soft_mode:
         self._soft_mode(logits, lists)
      else:
         self._hard_mode(logits, lists)

      return logits
   
   
class WaterMarkDetector:
   def __init__(self, cfg: WaterMarkConfig):

      assert cfg.detection_device is not None, \
         "Expected WatermarkConfig.detection_device to be set."
      assert isinstance(cfg.detection_device, str) or \
         isinstance(cfg.detection_device, torch.device), \
         "Expected detection_device to be of type torch.device or string."
      
      self.cfg = cfg
      self.generator = torch.Generator(self.cfg.detection_device)
      self.null_distribution = norm(0, 1)

   def _idInGreen(self, curr_id:int, seed_id:int, ):
      self.generator.manual_seed(self.cfg.hash_fn(seed_id))
      rand_perm = torch.randperm(n=self.cfg.vocab_size, 
                                 generator=self.generator,
                                 device=self.cfg.detection_device)
      # check if curr_id exist in the generated green list
      return (rand_perm[:self.cfg.G_list_size] == curr_id).sum() > 0

   @staticmethod
   def _zStatistic(s_G, T, gamma):
      num = s_G - (gamma * T)
      den = sqrt(T * gamma * (1-gamma))
      return num / den

   def detect(self, prompt: str, generation:str, tokenizer: AutoTokenizer):
      '''
      H0: the sequence has no knowledge of the red list rule.

      if z_stat < 4:
      then generated sequence has no knowledge of the red list rule.
      i.e. the sequence is either not watermarked, or human generated.
      AND cannot reject null huypothesis.
      detect method returns result:False 

      if z_stat > 4:
      REJECT numm hypothesis
      the generated sequence has knowledge of the red list rule.
      i.e. the sequence is watermarked
      detect method returns result:True 
      '''
      prompt_ids = tokenizer.encode(prompt)
      gen_id = tokenizer.encode(generation, add_special_tokens=False)

      num_gen_tokens = len(gen_id)
      detection_stats = []
      
      prev_id = prompt_ids[-1]
      s_G = 0 # maintain count of tokens generated from Green list
      for ix in range(num_gen_tokens):
         T = ix + 1 # length of sequence till this index
         curr_id = gen_id[ix]
         
         if self._idInGreen(curr_id, prev_id):
            s_G += 1

         z_stat = self._zStatistic(s_G, T, self.cfg.gamma)
         p_val = 1 - self.null_distribution.cdf(z_stat)
         result = z_stat > self.cfg.threshold
         detection_stats.append({'index': ix, 
                                 'z_stat': z_stat,
                                 'p_val': p_val,
                                 's_g': s_G, 
                                 'T':T, 
                                 'result':result,
                                 'token_id':curr_id})

         prev_id = curr_id
      
      return detection_stats