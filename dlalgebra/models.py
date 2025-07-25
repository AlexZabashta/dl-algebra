
import torch
import torch.nn as nn
    

class Vec2Vec(nn.Module):

    def __init__(self, input_adt, output_adt, torch_model):
        super().__init__()
        self.input_adt = input_adt
        self.output_adt = output_adt
        self.torch_model = torch_model

    def forward(self, x):
        return self.torch_model(x) 
    
    
class Eps2Vec(nn.Module):

    def __init__(self, input_adt, output_adt):
        super().__init__()
        self.input_adt = input_adt
        self.output_adt = output_adt
        self.param = torch.nn.Parameter(torch.zeros(output_adt.size))
    
    def forward(self, _):
        return self.param


class Prod2Vec(nn.Module):

    def __init__(self, input_adt, output_adt, models, kind='sum'):
        super().__init__()

        self.input_adt = input_adt
        self.output_adt = output_adt
        self.kind = kind
        
        self.models = nn.ModuleList(models)    
    
    def forward(self, list_x):
        out = []
        
        for model, x in zip(self.models, list_x):
            out.append(model(x))     
        
        if (self.kind == 'sum'):
            return torch.stack(out, dim=0).sum(dim=0)
        
        if (self.kind == 'concat'):
            return torch.cat(out)


class Sum2Vec(nn.Module):

    def __init__(self, input_adt, output_adt, models):
        super().__init__()

        self.input_adt = input_adt
        self.output_adt = output_adt
        
        self.models = nn.ModuleDict(models)    
    
    def forward(self, pair):
        adt_name, x = pair
        model = self.models[adt_name]
        return model(x)
        
