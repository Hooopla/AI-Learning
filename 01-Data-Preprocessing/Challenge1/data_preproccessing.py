import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

# Loading Iris Data Set 
dataset = pd.read_csv('iris.csv')

# Creating the matrix features (X) & dependant variable factor (y)
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1].values 

# Printing the values 
print("Independant Variables")
print(X)

print("Dependant Varibles")
print(y)