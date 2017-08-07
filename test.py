import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import Normalizer


def get_stock_data(stock_name, normalized=0):
    print("GETTING STOCK DATA")
    url="http://www.google.com/finance/historical?q="+stock_name+"&startdate=Jul+12%2C+2013&enddate=Jul+11%2C+2017&num=30&ei=rCtlWZGSFN3KsQHwrqWQCw&output=csv"

    col_names = ['Date','Open','High','Low','Close','Volume']
    stocks = pd.read_csv(url, header=0, names=col_names)
    df = pd.DataFrame(stocks)
    df.drop(df.columns[[0,3,5]], axis=1, inplace=True)
    return df

def load_data(stock, seq_len, file, train_percent=.75):

    data = stock.as_matrix()

    #iterate so that we can also capture a sequence for a target
    sequence_length = seq_len + 1

    print("SPLITTING INTO TIMESERIES")
    #segment the data into timeseries (these will be overlapping)
    result = []
    for index in range(len(data) - sequence_length):
        time_series = data[index: index + sequence_length]
        result.append(time_series)

    result = np.array(result)

    #train test split
    row = round(train_percent * result.shape[0])
    train = result[:int(row), :]
    test = result[int(row):, :]

    x_train = train[:, :-1]
    y_train = train[:, -1, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1, -1]

    for i in range(0, len(x_train)):
        for s in range(0, seq_len):
            file.write(str(x_train[i][s]) + "\n")
        file.write("Test: " + str(y_train[i]) + "\n")
        file.write("\n")

    for i in range(0, len(x_test)):
        for s in range(0, seq_len):
            file.write(str(x_test[i][s]) + "\n")
        file.write("Test: " + str(y_test[i]) + "\n")
        file.write("\n")

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


#MAIN()
file = open("rnn_output.txt", "w")

stock_name = 'GOOGL'
df = get_stock_data(stock_name,0)

# df['High'] = df['High'] / 1000
# df['Open'] = df['Open'] / 1000
# df['Close'] = df['Close'] / 1000

window = 5
X_train, y_train, X_test, y_test = load_data(df[::-1], window, file, train_percent=.9)

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

model = build_model([3,window,1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=500,
    validation_split=0.1,
    verbose=100)

trainScore = model.evaluate(X_train, y_train, verbose=100)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=100)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

p = model.predict(X_test)

for i in range(0, len(X_test)):
    for s in range(0, 5):
        file.write(str(X_test[i][s]) + "\n")
    file.write("Test: " + str(y_test[i]) + "\n")
    file.write("Prediction: " + str(p[i]) + "\n")
    file.write("\n")


plt.plot(p,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')
plt.legend(loc='upper left')
plt.show()