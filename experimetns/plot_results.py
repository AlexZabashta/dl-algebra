
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
from math import sqrt

directory = ".."

patterns = ["missing_25_result", "missing_50_result", "missing_75_result", "quantile_25_result", "quantile_50_result", "quantile_75_result"]
#patterns = [ "quantile_25_result", "quantile_50_result", "quantile_75_result"]

for pattern in patterns:
    ls = []    
    for sub_directory in os.scandir(directory):
        if (not pattern in sub_directory.name):
            continue
        s = set()
        for csv_file in os.scandir(sub_directory):
            s.add(csv_file.name)
        
        ls.append(s)
    
    si = None        
    for s in ls:
        if (si == None):
            si = s
        else:
            si = si.intersection(s)
    
    for sub_directory in os.scandir(directory):
        if (not pattern in sub_directory.name):
            continue
        
        list_d = []
        for csv_file in os.scandir(sub_directory):
            if (not csv_file.name in si):
                continue
            
            df = pd.read_csv(csv_file, header=None, names=['f'])
            list_f = list(df['f'].values)
            assert (len(list_f) == 50)
            list_d.append(list_f)            
    
        f_history = np.array(list_d)     
        f_history_mean = np.mean(f_history, axis=0)
        f_history_std = np.std(f_history, axis=0) / sqrt(459)
        
        label = sub_directory.name.replace(pattern, "").replace("_", "")
        if (label == ""):
            label = "ADTModel"
            
        plt.fill_between(range(50), np.subtract(f_history_mean, f_history_std), np.add(f_history_mean, f_history_std), alpha=0.2)
        plt.plot(f_history_mean, label=label)
        
    plt.legend()
    plt.title(pattern.replace("_result", "").replace("missing", "random"))
    plt.show()
