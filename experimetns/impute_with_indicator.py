import os 
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

for tmising in ["quantile"]:  #"missing", "quantile"
    for p_missing in [75]:
        directory_in = "..\\" + tmising + "_" + str(p_missing) 
        directory_out = "..\\" + "IterativeImputer" + "WI_" + tmising + "_" + str(p_missing) + "\\"
        
        if not os.path.exists(directory_out):
            os.makedirs(directory_out)
        
        for csv_file in os.scandir(directory_in):
            df = pd.read_csv(csv_file)
            target_column = df.columns[-1]
            X = df.drop(columns=[target_column])
            imp = IterativeImputer(keep_empty_features=True, add_indicator=True)
            X_imp = imp.fit_transform(X)
            _, n_features = X_imp.shape
            
            feature_names = imp.get_feature_names_out()
            #feature_names = list(map(lambda i: "feature" + str(i), range(n_features)))
                       
            
            df_imp = pd.DataFrame()
            df_imp[feature_names] = X_imp
            
            df_imp[target_column] = df[target_column]
            
            df_imp.to_csv(directory_out + csv_file.name, index=False)
            print(csv_file)

