import torch
from torch import nn
import os 
import pandas as pd
import numpy as np
import random
from math import sqrt

from dlalgebra.constructors import TSum2Vec as tsum
from dlalgebra.constructors import TProd2Vec as tprod
from dlalgebra.constructors import Vect as vect
from dlalgebra.constructors import Eps as eps
from dlalgebra.constructors import Apply as apply
from dlalgebra.model import SingleEmbedding

from sklearn.metrics import f1_score
dtype = torch.float32

directory_in = "..\\missing_75"
directory_out = directory_in + "_result"

if not os.path.exists(directory_out):
    os.makedirs(directory_out)

list_d = []

for csv_file in os.scandir(directory_in):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)   
    
    df = pd.read_csv(csv_file)
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    Y = pd.get_dummies(df[target_column], dtype=float)
    n_objects, n_features = X.shape
    n_classes = Y.shape[1]   
    
    s = (n_classes + n_features)    
    h_layer_size = max(3, round((sqrt(s * s + 4 * n_objects / 10) - s) / 2))
    
    print(csv_file, n_objects, n_features, n_classes, h_layer_size)
    
    weak_features_list = []
    for column in X:
        column = 'feature_' + column.replace('.', '_') 
        
        terms = {
            eps(): SingleEmbedding(h_layer_size),
            vect(adt_name=column, size=1): nn.Linear(1, h_layer_size)
            }
        weak_feature = tsum(adt_name='weak_' + column, terms=terms, size=h_layer_size)
        weak_features_list.append(weak_feature)
    
    n_classes = Y.shape[1]
    
    features = tprod(adt_name='features', kind='sum', terms=weak_features_list)
    
    f = nn.Sequential(nn.ReLU(), nn.Linear(h_layer_size, h_layer_size), nn.ReLU(), nn.Linear(h_layer_size, n_classes))
    
    model = apply(f, features, adt_name='model', size=n_classes)
    
    dataset = []

    for X_row, Y_row in zip(X.iterrows(), Y.iterrows()):
        X_row = X_row[1]
        Y_row = Y_row[1]
    
        x = list()    
        for column in X.columns:
            v = X_row[column]
            
            column = 'feature_' + column.replace('.', '_')                
            if (np.isnan(v)):
                f = ('eps', None)
            else:
                f = (column, torch.tensor([v], dtype=dtype))
            x.append(f)
            
        y = torch.tensor(Y_row.values, dtype=dtype)
        
        obj = (x, y)
        dataset.append(obj)
    
    random.shuffle(dataset)

    train_part = round(n_objects * 0.75)
    train = dataset[:train_part]
    test = dataset[train_part:]
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    list_f = []
    
    for _ in range(50):
        
        model.train()
        random.shuffle(train)
        for x, y in train:
            optimizer.zero_grad()
            out = model(x)
            l = loss(out, y)
            l.backward()
            optimizer.step()            
        
        model.eval()
        
        y_pred = []
        y_true = []
        
        for x, y in test: 
            out = model(x)        
            y_pred.append(torch.argmax(out).item())
            y_true.append(torch.argmax(y).item())
        
        list_f.append(f1_score(y_true, y_pred, average='weighted'))
        
        # print()
    with open(directory_out + "\\" + csv_file.name, "w") as f:
        for val in list_f:
            f.write(str(val))
            f.write('\n')

