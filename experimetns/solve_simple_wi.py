import torch
from torch import nn
import os 
import pandas as pd
import numpy as np
import random
from math import sqrt

from sklearn.metrics import f1_score
dtype = torch.float32

# SimpleImputer, KNNImputer, IterativeImputer
# 25 50 75
# quantile missing

method = "SimpleImputer" + "WI"
miss_type = "missing"
miss_persent = "75"

directory_in = "..\\" + method + "_" + miss_type + "_" + miss_persent
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
    
    s = (n_classes + n_features / 2)    
    h_layer_size = max(3, round((sqrt(s * s + 4 * n_objects / 10) - s) / 2))
    
    print(csv_file, n_objects, n_features, n_classes, h_layer_size)
    
    model = nn.Sequential(nn.Linear(n_features, h_layer_size), nn.ReLU(), nn.Linear(h_layer_size, h_layer_size), nn.ReLU(), nn.Linear(h_layer_size, n_classes))
    
    dataset = []

    for i in range(n_objects): 
        x = torch.tensor(X.iloc[i].values, dtype=dtype)        
        y = torch.tensor(Y.iloc[i].values, dtype=dtype) 
        
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
     
