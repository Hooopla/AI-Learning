import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
from sklearn.impute import SimpleImputer # type: ignore 

# Loading CSV
dataset = pd.read_csv('iris.csv')

# Loading Dependant and Independant Variables 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[: , -1].values 

# Taking care of missing data 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Categorical Data 
print(X)
print(y)