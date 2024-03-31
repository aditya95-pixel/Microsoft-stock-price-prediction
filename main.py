from datetime import datetime 
import tensorflow as tf 
from tensorflow import keras 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
import numpy as np 
import seaborn as sns 
microsoft = pd.read_csv('MicrosoftStock.csv') 
print(microsoft.head()) 
print(microsoft.shape)
print(microsoft.info())
microsoft.describe()
plt.plot(microsoft['date'], 
         microsoft['open'], 
         color="blue", 
         label="open") 
plt.plot(microsoft['date'], 
         microsoft['close'], 
         color="green", 
         label="close") 
plt.title("Microsoft Open-Close Stock") 
plt.legend() 
plt.plot(microsoft['date'], 
         microsoft['volume']) 
plt.show()
numeric_columns = microsoft.select_dtypes(include=np.number)
correlation_matrix = numeric_columns.corr()
# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cbar=False)
plt.show() 
microsoft['date'] = pd.to_datetime(microsoft['date']) 
prediction = microsoft.loc[(microsoft['date'] 
							> datetime(2013, 1, 1)) 
							& (microsoft['date'] 
							< datetime(2018, 1, 1))] 
plt.figure(figsize=(10, 10)) 
plt.plot(microsoft['date'], microsoft['close']) 
plt.xlabel("Date") 
plt.ylabel("Close") 
plt.title("Microsoft Stock Prices") 
plt.show()
# prepare the training set samples 
msft_close = microsoft.filter(['close']) 
dataset = msft_close.values 
training = int(np.ceil(len(dataset) *.95)) 
# scale the data 
ss = StandardScaler() 
ss = ss.fit_transform(dataset) 
train_data = ss[0:int(training), :] 
x_train = [] 
y_train = [] 
# considering 60 as the batch size, 
# create the X_train and y_train 
for i in range(60, len(train_data)): 
	x_train.append(train_data[i-60:i, 0]) 
	y_train.append(train_data[i, 0]) 
x_train, y_train = np.array(x_train),np.array(y_train) 
X_train = np.reshape(x_train, 
					(x_train.shape[0], 
					x_train.shape[1], 1)) 
model = keras.models.Sequential() 
model.add(keras.layers.LSTM(units=64, 
							return_sequences=True, 
							input_shape 
							=(X_train.shape[1], 1))) 
model.add(keras.layers.LSTM(units=64)) 
model.add(keras.layers.Dense(128)) 
model.add(keras.layers.Dropout(0.5)) 
model.add(keras.layers.Dense(1)) 
print(model.summary()) 
from keras.metrics import RootMeanSquaredError 
rmse = RootMeanSquaredError()
model.compile(optimizer='adam', 
			loss='mae', 
			metrics=[rmse]) 
history = model.fit(X_train, y_train, 
					epochs=20) 
testing = ss[training - 60:, :] 
x_test = [] 
y_test = dataset[training:, :] 
for i in range(60, len(testing)): 
	x_test.append(testing[i-60:i, 0]) 
x_test = np.array(x_test) 
X_test = np.reshape(x_test, 
					(x_test.shape[0], 
					x_test.shape[1], 1)) 
pred = model.predict(X_test) 
train = microsoft[:training] 
test = microsoft[training:] 
test = test.copy()  # Make an explicit copy to avoid chained assignment warning
test.loc[:, 'Predictions'] = pred
plt.figure(figsize=(10, 8)) 
plt.plot(train['close'], c="b") 
plt.plot(test[['close', 'Predictions']]) 
plt.title('Microsoft Stock Close Price') 
plt.ylabel("Close") 
plt.legend(['Train', 'Test', 'Predictions']) 
plt.show()
from sklearn.metrics import mean_squared_error
# Calculate RMSE for the test set predictions
rmse_test = np.sqrt(mean_squared_error(y_test, pred))
print("RMSE on Test Set:", rmse_test)
model.save("model.keras")