# Microsoft Stock Price Prediction
This Python script utilizes LSTM (Long Short-Term Memory) neural networks to predict the closing prices of Microsoft stock based on historical data. Below is an overview of how the script works:

## 1. Data Loading and Visualization
The script imports necessary libraries such as TensorFlow, pandas, matplotlib, and seaborn.

It loads historical Microsoft stock price data from a CSV file named MicrosoftStock.csv into a pandas DataFrame.

Initial exploration of the dataset is performed, including displaying the first few rows, shape, and information about the columns.

Visualizations are created to show the trends in opening and closing prices, as well as trading volume over time. Additionally, a heatmap is generated to visualize the correlation between different numerical features in the dataset.

## 2. Data Preprocessing
The 'date' column is converted to datetime format to facilitate time-based analysis.

The dataset is split into a training set and a testing set. The training set comprises 95% of the data, and the testing set contains the remaining 5%.

Data is scaled using StandardScaler to normalize it for improved model training performance.

## 3. Model Building and Training

An LSTM neural network model is constructed using the Keras API. The model architecture includes two LSTM layers with 64 units each, followed by a Dense layer with 128 units and a dropout layer to prevent overfitting. Finally, a Dense layer with one unit is added for regression.

The model is compiled with the Adam optimizer, Mean Absolute Error (MAE) loss function, and Root Mean Squared Error (RMSE) as the evaluation metric.

The model is trained on the training data for 20 epochs.

## 4. Model Evaluation and Prediction

The trained model is used to make predictions on the testing set.

The predicted closing prices are compared with the actual closing prices to evaluate the model's performance using RMSE.

The script saves the trained model as a .keras file for future use.
