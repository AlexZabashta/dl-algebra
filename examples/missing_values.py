
from dlalgebra.adt_constructors import TProd, TSum, Vec
from dlalgebra.static_builders import build_type2vec
from sklearn.datasets import load_iris
import numpy as np
import random
import torch
from torch import nn
from sklearn.metrics import f1_score

np.random.seed(0) 
random.seed(0)

dtype = torch.float32
h_layer_size = 2


def add_missing_values(X, p_missing=0.5):
    n_objects, n_features = X.shape
        
    n_values = n_objects * n_features
    n_missing = round(n_values * p_missing)
    
    mask = ([True] * n_missing) + ([False] * (n_values - n_missing))
    random.shuffle(mask)
    
    mask2D = np.reshape(mask, shape=(n_objects, n_features)) 
    
    X[mask2D] = np.nan      


def main():
    
    iris = load_iris()
    
    X = iris.data
    n_objects, n_features = X.shape
    
    n_classes = len(iris.target_names)
    Y = iris.target
    
    add_missing_values(X)
    
    dataset = []

    for X_row, Y_row in zip(X, Y):
        x = list()    
        for i, feature_name in enumerate(iris.feature_names):
            v = X_row[i]
            
            feature_name = 'feature_' + feature_name.replace('.', '_').replace(' ', '_')                
            if (np.isnan(v)):
                f = ('missing', None)
            else:
                f = (feature_name, torch.tensor([v], dtype=dtype))
            x.append(f)
            
        y = torch.tensor(Y_row, dtype=torch.long)
        dataset.append((x, y))
    
    random.shuffle(dataset)

    features_types = []
    
    for feature_name in iris.feature_names:
        feature_name = 'feature_' + feature_name.replace('.', '_').replace(' ', '_')
        
        feature_types = []
        
        feature_types.append(Vec("missing", 0))
        feature_types.append(Vec(feature_name, 1))
            
        weak_feature_type = TSum("weak_" + feature_name, *feature_types)
        
        features_types.append(weak_feature_type)
    
    X_type = TProd("features", *features_types)
    # Y_type = Vec("target", n_classes)

    H_type = Vec("hidden", h_layer_size)
    
    model = build_type2vec(X_type, H_type, dict())
    
    model = nn.Sequential(model, nn.ReLU(), nn.Linear(h_layer_size, n_classes))
    
    train_part = round(n_objects * 0.75)
    train = dataset[:train_part]
    test = dataset[train_part:]
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for _ in range(150):
        
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
            y_true.append(y.item())
        
        print((f1_score(y_true, y_pred, average='weighted')))


if __name__ == '__main__':
    main()

