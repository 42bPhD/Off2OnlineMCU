from fxpmath import Fxp
import numpy as np
from utils.utils import reproducibility
from time import time
def input_format(size:tuple, bit=8, seed = time()):
    reproducibility(int(seed))
    tensor = np.random.random(size)
    tensor = Fxp(tensor, signed=True, n_word=bit, overflow='saturate')
    tensor.config.dtype_notation = 'Q'
    tensor.config.array_output_type = 'array'
    frac = tensor.n_frac
    return (tensor, tensor.val , frac)