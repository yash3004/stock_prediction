#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:07:40 2022

@author: yashsehgal
"""

#importing the libs
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#importing the dataset
df = pd.read_csv("CSVForDate.csv")
df2 = pd.read_csv("sensex.csv")
df2 = df2.dropna(how ='all' , axis = 1)
df2 = df2.dropna()
#checking for the null values
print(df.isnull().any(axis=0))
#removing the null values
df1 = df.dropna(how = 'all',axis = 1)
df1 = df1.dropna()
#dataset has been created
stc_price = df1.iloc[: ,0:1].values
#scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
stc_price = sc.fit_transform(stc_price)

#creating a dsa with 60 time steps
x_train = []
y_train = []
for i in range(60,7300):
    x_train.append(stc_price[i-60:i,0])
    y_train.append(stc_price[i,0])
x_train , y_train = np.array(x_train) , np.array(y_train)
#reshaping the data
x_train = np.reshape(x_train , (x_train.shape[0],x_train.shape[1] , 1))
#building the rnn 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
#init the model 
model = Sequential()
model.add(LSTM(units = 50 , return_sequences =True , input_shape = (x_train.shape[1] , 1)))
model.add(Dropout(0.2))
#adding another layer 
model.add(LSTM(units = 50 , return_sequences =True ))
model.add(Dropout(0.2))
#adding another layer 
model.add(LSTM(units = 50 , return_sequences =True ))
model.add(Dropout(0.2))
#adding another layer 
model.add(LSTM(units = 50  ))
model.add(Dropout(0.2))
#adding output layer
model.add(Dense(units = 1))
#compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#fitting the data into it
model.fit(x_train ,  y_train , epochs = 10, batch_size = 32)
#predicting the values from
stc_price2 = df2.iloc[: , 0:1].values
df_total = pd.concat((df1['Open'] , df2['Open']) , axis = 0)
inputs = df_total[len(df_total) - len(df2) - 60:].values
inputs = inputs.reshape(-1 , 1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60 ,576):
    x_test.append(inputs[i-60:i , 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test ,(x_test.shape[0] , x_test.shape[1] , 1))
predicted = model.predict(x_test)
predicted = sc.inverse_transform(predicted)
#visualising 
plt.plot(stc_price2 , color = 'red')
plt.plot(predicted , color = 'blue')
plt.show()