import torch
from torch import nn

import pandas as pd
import numpy as np
import random
from sklearn.metrics import f1_score
dtype = torch.float32

df = pd.read_csv('..\\datasets\\missing_rnd.csv')

target_column = 'variety'

X = df.drop(columns=[target_column])

Y = pd.get_dummies(df[target_column], dtype=float)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = IterativeImputer()

X = imp.fit_transform(X)

weak_features_list = []
n_classes = Y.shape[1]
h_layer_size = 5

n_features = X.shape[1]

model = nn.Sequential(nn.Linear(n_features, h_layer_size), nn.ReLU(), nn.Linear(h_layer_size, n_classes))

dataset = []

for Y_row in Y.iterrows():
    X_row = X[Y_row[0]]
    Y_row = Y_row[1]

    x = torch.tensor(X_row, dtype=dtype)        
    y = torch.tensor(Y_row.values, dtype=dtype)
    
    obj = (x, y)
    dataset.append(obj)
    
random.shuffle(dataset)

train = dataset[:120]
test = dataset[120:]

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for _ in range(100):
    
    model.train()
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
    
    print(f1_score(y_true, y_pred, average='weighted'))

    # print(obj)

# encode type 

# encode dataset

# F0 = tsum(eps(),)

# X = tprod(_, eps)

# print(df)

# print(x)
# print(y)

# print(Linear(1, 3))

