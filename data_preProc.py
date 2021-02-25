import numpy as np 
import matplotlib.pyplot as plt     
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing dataset
dataset = pd.read_csv('/home/pranavan/Downloads/Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv') 
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

# Dealing with missing values in dataset
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #this gives the mean value of respective columns to the missing values.
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)
print(y)

# Encoding Independent Variables
# Because the dataset has string values, these need to be convertred to numbers using encoders. For categorical data we use OneHotEncoder.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')#transformer=[(transform_type, class_used_for_encoding, [index_of_column])], remainder= 
x = np.array(ct.fit_transform(x))
print(x)

# Encoding Dependent Variable 
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting training and testing dataset 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print('x_train: ', x_train)
print('x_test: ', x_test)
print('y_train: ', y_train)
print('y_test: ', y_test)

# Feature  Scalling 
# Put all the features on the same scale. Standardization[-3 : 3], Normalization [0 : 1]
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print('scalled x_train: ', x_train)
print('scalled x_test: ', x_test)