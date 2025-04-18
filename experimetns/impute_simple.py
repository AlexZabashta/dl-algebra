import os 
import pandas as pd
from sklearn.impute import SimpleImputer

for tmising in ["quantile", "missing"]:
    for p_missing in [25, 50, 75]:
        directory_in = "..\\" + tmising + "_" + str(p_missing) 
        directory_out = "..\\SimpleImputer_" + tmising + "_" + str(p_missing) + "\\"
        
        if not os.path.exists(directory_out):
            os.makedirs(directory_out)
        
        for csv_file in os.scandir(directory_in):
            df = pd.read_csv(csv_file)
            target_column = df.columns[-1]
            X = df.drop(columns=[target_column])
            df[X.columns] = SimpleImputer(keep_empty_features=True).fit_transform(X)
            df.to_csv(directory_out + csv_file.name, index=False)

