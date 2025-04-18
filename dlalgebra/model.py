
import torch
import torch.nn as nn


class Model:

    def __init__(self, from_type, to_type):
        pass
    
    def forward(self):
        pass
    
    
class SingleEmbedding(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(size))
    
    def forward(self, _):
        return self.param
