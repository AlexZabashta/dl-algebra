import pandas as pd
import numpy as np
import random
import os

directory = "..\\csv"
for csv_file in os.scandir(directory):
    for p_missing in [25, 50, 75]:
        np.random.seed(0) 
        random.seed(0)
        
        df = pd.read_csv(csv_file)
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        
        X = (X - X.mean()) / X.std()
        
        y = df[target_column]
        
        n_objects, n_features = X.shape
        
        n_classes = len(set(y))
        v = X.to_numpy()
        
        n_values = n_objects * n_features
        n_missing = round(n_values * p_missing / 100)      
        
        mask = ([True] * n_missing) + ([False] * (n_values - n_missing))
        random.shuffle(mask)
        vf = v.flatten()
        vf[mask] = np.nan
        v = np.reshape(vf, shape=(n_objects, n_features))
        
        mdf = pd.DataFrame()
        mdf[X.columns] = v
        mdf[target_column] = y
        
        mdf.to_csv("..\\missing_" + str(p_missing) + "\\" + csv_file.name, index=False)
      
