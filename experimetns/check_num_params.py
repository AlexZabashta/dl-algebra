from math import sqrt, ceil
import random
import os
import pandas as pd
from math import sqrt, ceil

if __name__ == '__main__':
    directory_in = "..\\..\\part_missing"
    for csv_file in os.scandir(directory_in):
        df = pd.read_csv(csv_file)
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        n_objects, n_features = X.shape
        
        n_classes = len(set(y))
        
        s = (n_classes + n_features)
    
        h = (sqrt(s * s + 4 * n_objects / 10) - s) / 2
        h = max(round(h), 3)
        
        n_params = n_features * h + h * h + h * n_classes
        
        n_mean = round(sqrt(n_classes * n_features))
        h = n_mean 
        
        n_params2 = n_features * h + h * h + h * n_classes
        
        print(n_objects, n_features, n_classes, n_params, n_params2)  
    
