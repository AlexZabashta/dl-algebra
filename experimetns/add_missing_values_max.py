import pandas as pd
import numpy as np
import os

directory = "..\\csv"
for csv_file in os.scandir(directory):
    for p_missing in [25, 50, 75]:
        
        df = pd.read_csv(csv_file)
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        
        X = (X - X.mean()) / X.std()
        
        y = df[target_column]
        
        q = X.quantile(p_missing / 100)
        X[X < q] = np.nan
        X[target_column] = df[target_column]        
        
        X.to_csv("..\\quantile_" + str(p_missing) + "\\" + csv_file.name, index=False)
