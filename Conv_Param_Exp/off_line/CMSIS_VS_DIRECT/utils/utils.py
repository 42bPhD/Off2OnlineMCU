from collections import defaultdict
import os

import torch
import numpy as np
def ddict():
    return defaultdict(ddict)


def list_bracket(lists):
    return "{" + ', '.join(map(str, lists)) + "}"


def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=np.inf) #extend numpy
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        

