"""
MLP actor for continous control.
"""

import torch.nn as nn
import numpy as np
import torch

class MLP_Actor(nn.Module):
    def __init__(self):
        