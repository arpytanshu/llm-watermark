
import torch
from typing import Callable


class WaterMarkingConfig:
    def __init__(self, 
                 vocab_size: int, 
                 gamma: float = 0.5, 
                 hardness: float = 1.5,
                 hash_fn: Callable = hash,
                 device: torch.device = torch.device('cpu'), 
                 soft_mode: bool = True):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.hardness = hardness # delta
        self.soft_mode = soft_mode
        self.device = device
        self.hash_fn = hash_fn

        # Calculate G and R list sizes
        self.G_list_size = int(self.vocab_size * self.gamma)
        self.R_list_size = self.vocab_size - self.G_list_size

    def __repr__(self):
        return \
         f"WaterMarkingConfig(vocab_size={self.vocab_size}, \
         gamma={self.gamma}, \
         hardness={self.hardness}, \
         soft_mode={self.soft_mode})"
    

class WaterMarking:
   def __init__(self, cfg: WaterMarkingConfig, hash_fn: Callable):
      self.cfg = cfg
      self.hash_fn = hash_fn
   
   def _hash_input(self, input_ids):
      '''
      input_ids: expected_shape is Bx1
      step 3 of Algorithm 2, which creates R list & G list.
      '''
      g = torch.Generator()
      bch_sz = input_ids.shape[0]
      G_list = torch.empty((bch_sz, self.cfg.G_list_size), dtype=torch.long)
      R_list = torch.empty((bch_sz, self.cfg.R_list_size), dtype=torch.long)
      
      for ix, id in enumerate(input_ids):
         g.manual_seed(self.hash_fn(id.item()))
         rand_perm = torch.randperm(n=self.cfg.vocab_size, generator=g)
         G_list[ix] = rand_perm[:self.cfg.G_list_size]
         R_list[ix] = rand_perm[self.cfg.G_list_size:]
      
      return {'R_list': R_list, 'G_list': G_list}
   
   def _hard_mode(self, logits, lists):
      logits.scatter_(index=lists['R_list'], dim=1, value=0)
   
   def _soft_mode(self, logits, lists):
      index = lists['G_list']
      src = torch.ones(index.shape) * self.cfg.hardness
      logits.scatter_reduce_(dim=1, index=index, src=src, reduce='sum')

   def __call__(self, logits: torch.FloatTensor, input_ids: torch.LongTensor, return_lists:bool = False):
      assert logits.dim() == 3, \
         f"Expected logits to have dimension 3, but got dimension {logits.dim()}"
      assert input_ids.dim() == 2, \
         f"Expected input_ids to have dimension 2, but got dimension {input_ids.dim()}"

      last_input_ids = input_ids[:, -1].view(-1, 1) # slice input_ids to be hashed. [b]
      lists = self._hash_input(last_input_ids)

      logits = logits[:, -1, :] # slice logits from last input token. [b x v]
      
      if self.cfg.soft_mode:
         self._soft_mode(logits, lists)
      else:
         self._hard_mode(logits, lists)

      if return_lists:
         return logits, lists
      else:
         return logits
      

