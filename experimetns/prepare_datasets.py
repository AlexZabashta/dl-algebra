import os
import pandas as pd
from scipy.io import arff
from sklearn.decomposition import PCA
from math import sqrt

directory = "..\\..\\arff"
directory_csv = "..\\..\\csv\\"  # set directory path

for arff_file in os.scandir(directory):
    try:
        a = arff.loadarff(arff_file)
        df = pd.DataFrame(a[0])
        if (df.dtypes.iloc[-1] != object):
            continue
        
        print(arff_file.name)
        
        target_column = df.columns[-1]
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        Xnum = X.select_dtypes(include='number')
        Xobj = X.select_dtypes(exclude='number')
        
        csv = pd.DataFrame()
        csv[Xnum.columns] = Xnum
        
        if (Xobj.shape[1] != 0):
            Xobj = pd.get_dummies(Xobj, dtype=float)
            n_features = Xobj.shape[1]
            n_comp = max(min(10, n_features), int(sqrt(n_features)))
            
            if (n_features == n_comp):
                csv[Xobj.columns] = Xobj
            else:
                pca_columns = list(map(lambda i: 'cat_pca_' + str(i), range(n_comp)))
                pca = PCA(n_components=n_comp)
                
                csv[pca_columns] = pca.fit_transform(Xobj)
                
        csv[target_column] = df[target_column] 
        
        csv.to_csv(directory_csv + arff_file.name.replace('arff', 'csv'), index=False)
                # print(arff_file, n_features, n_comp)
        
        # 
        
        
        # print(df.dtypes.iloc[-1] == object)
        # break
        
    except Exception as e:
        print(e)
