from .base import ESResNet
from .base import ESResNeXt
from .fbsp import ESResNetFBSP
from .fbsp import ESResNeXtFBSP
from .attention import Attention2d

import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')
__all__ = ['ESResNet', 'ESResNeXt', 'ESResNetFBSP', 'ESResNeXtFBSP', 'Attention2d']
