import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#  Import dataset:
dataset = pd.read_csv('/home/pranav/Documents/py_DEV/ML_A-Z/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train test split: 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Train training set: 
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict(y_pred) based on the test feature(x_test) results:
y_pred = regressor.predict(x_test)

# Plot training set results: 
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Plot test prediction results:
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Make single prediction:
print(regressor.predict([[12]]))# Double square bracket(list of a list) makes the value 2D because predict method only accepts 2D arrays  

# Get Linear Regression Equation coefficients:
print(regressor.coef_) 
print(regressor.intercept_)


