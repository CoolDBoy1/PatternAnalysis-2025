# modules.py
# This file will contain all model components (classes/functions)

import torch
import torch.nn as nn

# Example skeleton class
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        # define layers here
        pass

    def forward(self, x):
        # define forward pass
        return x