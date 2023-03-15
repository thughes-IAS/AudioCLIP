from .clip import *
from .esresnet import *
from .audioclip import AudioCLIP

import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')