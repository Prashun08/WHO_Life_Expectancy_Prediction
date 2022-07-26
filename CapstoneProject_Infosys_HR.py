# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:50:02 2022

@author: pchakrabor24
"""
#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

#Importting the dataset
data = pd.read_csv('HR_comma_sep.csv')

#Checking the characteristics of Data
data.describe()
data.info()

#Visualising Data
plot = sns.pairplot(data,hue='left')

#Using Normalisation for Average Monthly Hours
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(data[['average_montly_hours']])
data['Scaled_Monthly_Average_hours'] = scaled_values[:,:]

data.drop('average_montly_hours',axis=1,inplace=True)

#Defining Input Output
x = data.columns.drop('left')
y = data['left']
x = data[x]

#One hot Encoding
x_encoded = pd.get_dummies(x)

# Train Test Split Model
x_train, x_test, y_train, y_test = train_test_split(x_encoded,y,test_size=0.33)

#Using Support Vector Machine
from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)

#Checking the coefficiant of determination
model.score(x_train, y_train)
model.score(x_test, y_test)

#Creating Confusion Matrix
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix
train_conf_matrix = confusion_matrix(y_train, train_pred)
test_conf_matrix = confusion_matrix(y_test, test_pred)
print(train_conf_matrix)
print(test_conf_matrix)

#Projecting Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, test_pred))

