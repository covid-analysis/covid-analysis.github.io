# Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('toronto_corona.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

# Encoding categorical data VAR --> X
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
(X[:, 4]) = labelencoder_X.fit_transform(X[:, 4])

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Regression to the dataset 
from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 1)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Calculating Root Mean Squared Error
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))

# Calculating R2
from sklearn.metrics import r2_score
coefficient_of_dermination = r2_score(y_test, y_pred)
print (coefficient_of_dermination)

# Calculating Mean Absolute Percentage Error
from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_test, y_pred): 
    y_test, y_pred = check_arrays(y_test, y_pred)
    print (np.mean(np.abs((y_test - y_pred) / y_test)) * 100)