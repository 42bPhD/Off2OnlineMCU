from fxpmath import Fxp
import numpy as np
from utils.utils import reproducibility
from time import time
import torch
def input_format(size:tuple, bit=8):
    reproducibility(1234)
    tensor = np.random.random(size)
    tensor = Fxp(tensor, signed=True, n_word=bit, overflow='saturate')
    tensor.config.dtype_notation = 'Q'
    tensor.config.array_output_type = 'array'
    frac = tensor.n_frac
    return (tensor, tensor.val , frac)