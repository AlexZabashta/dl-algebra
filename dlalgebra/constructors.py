
import torch
import torch.nn as nn


class Apply(nn.Module):
    
    def __init__(self, tensor_function, adt_function, adt_name=None, size=None):
        super().__init__()
        if (adt_name == None):
            adt_name = 'apply' + str(id(self))
        
        self.adt_name = str(adt_name)
        self.size = size
        self.tensor_function = tensor_function
        self.adt_function = adt_function
        
    def forward(self, x):
        y = self.adt_function(x)        
        return self.tensor_function(y)
    
    def __str__(self):
        return self.adt_name


class TSum2Vec(nn.Module): 

    def __init__(self, adt_name=None, terms=None, size=None):
        super().__init__()
        if (adt_name == None):
            adt_name = 'sum' + str(id(self))
            
        self.adt_name = str(adt_name)
        
        tterms = {}
        tensor_functions = {}
        
        for term, module in terms.items():
            tensor_functions[term.adt_name] = module # .replace('.', '_')
            tterms[term.adt_name] = term
        
        self.terms = tterms
        self.tensor_functions = nn.ModuleDict(tensor_functions)  # dot in adt_name
        
        self.size = size
        
    def __str__(self):
        return self.adt_name
    
    def forward(self, x): 
        tadt_name, inp = x        
        term, tensor_function = self.terms[tadt_name], self.tensor_functions[tadt_name] #.replace('.', '_')
        
        return tensor_function(term(inp))
        
        
class TProd2Vec(nn.Module):

    def __init__(self, adt_name=None, kind='concat', terms=None):
        super().__init__()
        if (adt_name == None):
            adt_name = 'prod' + str(id(self))
            
        self.adt_name = str(adt_name)
        
        assert (kind in ['concat', 'sum']), 'wrong kind = ' + str(kind)        
        
        self.kind = kind
        
        self.terms = nn.ModuleList(terms)
        
        size = None
        
        if (kind == 'concat'):
            size = 0
            for term in terms:
                size += term.size
        else:
            for term in terms:
                if (size == None):
                    size = term.size
                else:
                    assert (size == term.size)
            
            assert (size != None)
        
        self.size = size
        
    def __str__(self):
        return self.adt_name
    
    def forward(self, x):
        out = []
        
        for tensor_function, inp in zip(self.terms, x):
            out.append(tensor_function(inp))     
        
        if (self.kind == 'sum'):
            return torch.stack(out, dim=0).sum(dim=0)
        
        if (self.kind == 'concat'):
            return torch.cat(out)
   

class Eps(nn.Module):

    def __init__(self):
        super().__init__()
        self.adt_name = 'eps'
        self.size = 0
    
    def forward(self, _):
        return torch.tensor([])
    
    def __str__(self):
        return str(type(self))


class Vect(nn.Module):

    def __init__(self, adt_name=None, size=None):
        super().__init__()
        if (adt_name == None):
            adt_name = 'vect' + str(id(self))
        
        self.adt_name = str(adt_name)
        self.size = size
        
    def __str__(self):
        return self.adt_name
    
    def forward(self, x):
        return x

