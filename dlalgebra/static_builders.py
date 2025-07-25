
import torch
import torch.nn as nn

from dlalgebra.adt_constructors import TProd, TSum, Vec
from dlalgebra.models import Prod2Vec, Sum2Vec, Eps2Vec, Vec2Vec


def build_prod2vec(prod_adt, vec_adt, models_dict, kind='sum'):
    key = (prod_adt.name, vec_adt.name)
    if (key in models_dict):
        model = models_dict[key]
        if (model == "LOOP"):
            raise Exception("Recursive definition")    
        return model
    models_dict[key] = "LOOP"
    
    kind = str(kind).lower()
    
    if (kind == 'sum'):
        auxiliary_funcions = []
        for term in prod_adt.terms:
            f = build_type2vec(term, vec_adt, models_dict)
            auxiliary_funcions.append(f)
        model = Prod2Vec(prod_adt, vec_adt, auxiliary_funcions, kind)
        models_dict[key] = model
        return model
    
    raise Exception("Wrong kind " + kind)


def build_sum2vec(sum_adt, vec_adt, models_dict):
    key = (sum_adt, vec_adt)
    if (key in models_dict):
        model = models_dict[key]
        if (model == "LOOP"):
            raise Exception("Recursive definition")    
        return model
    models_dict[key] = "LOOP"
    
    auxiliary_funcions = {}
    for term in sum_adt.terms:
        f = build_type2vec(term, vec_adt, models_dict)
        auxiliary_funcions[term.name] = f
    
    model = Sum2Vec(sum_adt, vec_adt, auxiliary_funcions)
    models_dict[key] = model
    return model


def build_vec2vec(input_adt, output_adt, models_dict):
    key = (input_adt.name, output_adt.name)
    
    if (key in models_dict):
        model = models_dict[key]
        
        if (type(model) != Vec2Vec and type(model) != Eps2Vec):
            model = Vec2Vec(input_adt, output_adt, model)
            models_dict[key] = model

        return model
    
    if (input_adt.size == 0):
        model = Eps2Vec(input_adt, output_adt)
    else:
        model = Vec2Vec(input_adt, output_adt, nn.Linear(input_adt.size, output_adt.size))
    
    models_dict[key] = model
    return model  


def build_type2vec(input_adt, output_adt, models_dict):
    if (type(output_adt) != Vec):
        raise Exception("Wrong output type " + str(output_adt))
    
    if (type(input_adt) == Vec):
        return build_vec2vec(input_adt, output_adt, models_dict)
    
    if (type(input_adt) == TProd):
        return build_prod2vec(input_adt, output_adt, models_dict)
    
    if (type(input_adt) == TSum):
        return build_sum2vec(input_adt, output_adt, models_dict)
    
    raise Exception("Wrong input type " + str(input_adt))

