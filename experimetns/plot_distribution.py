
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

directory = "..\\..\\csv"


list_n_objects = []
list_n_features = []
list_n_classes = []


for csv_file in os.scandir(directory):
    df = pd.read_csv(csv_file)
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    Y = pd.get_dummies(df[target_column], dtype=float)
    n_objects, n_features = X.shape
    n_classes = Y.shape[1]   
    
    list_n_objects.append(n_objects)
    list_n_features.append(n_features)
    list_n_classes.append(n_classes)
 

print(list_n_objects)
print(list_n_features)
print(list_n_classes)

# Plotting a basic histogram
plt.hist(list_n_objects, bins=20) 
plt.title('n_objects distribution') 
# Display the plot
plt.show()

plt.hist(list_n_features, bins=20) 
plt.title('n_features distribution') 
# Display the plot
plt.show()

plt.hist(list_n_classes, bins=20) 
plt.title('n_classes distribution') 
# Display the plot
plt.show()

#plt.legend()
#plt.title(pattern.replace("_result", "").replace("missing", "random"))
#plt.show()
