#!/usr/bin/env python3

import os
import sys
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

CONV2D = nn.Conv2d
MAXPOOL = nn.MaxPool2d

X = torch.randn(1, 3, 224, 224)
if __name__ == '__main__':
    pass