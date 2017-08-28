import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multiple__targets.model_mult import build_model


#gets stock data from google finance, excludes some columns, then returns a dataframe
def get_stock_data(stock_name):
    print("GETTING STOCK DATA")
    url = "http://www.google.com/finance/historical?q=" + stock_name + "&startdate=Jul+10%2C+2010&enddate=Jul+27%2C+2017&num=30&ei=rCtlWZGSFN3KsQHwrqWQCw&output=csv"

    col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    stocks = pd.read_csv(url, header=0, names=col_names)
    df = pd.DataFrame(stocks)
    df.drop(df.columns[[0, 3, 5]], axis=1, inplace=True)

    return df




#Loads in stock data from a dataframe and tra
def load_data(stock, seq_len, target_len, train_percent=.75):
    data = stock.as_matrix()

    # iterate so that we can also capture a sequence for a target
    combined_length = seq_len + target_len

    print("SPLITTING INTO TIMESERIES")

    # segment the data into timeseries (these will be overlapping)
    result = []
    for index in range(len(data) - combined_length):
        time_series = data[index: index + combined_length]
        result.append(time_series[:])

    # normalize
    reference_points = [] # for de-normalizing outside of the function
    for i in range(0, len(result)):
        reference_points.append(result[i][0][0])
        result[i] = (((result[i]) / (result[i][0][0])) - 1)

    result = np.asarray(result)

    # train test split
    row = round(train_percent * result.shape[0])
    train = result[:int(row), :]
    test = result[int(row):, :]

    x_train = train[:, :target_len]
    y_train = train[:, target_len:, -1]

    x_test = test[:, :target_len]
    y_test = test[:, target_len:, -1]


    #In case we want to train on percent increase rather than a stock value
    # for i in range(0, len(train_target_timeseries)):
    #     start_price = x_train[i][-1][2]
    #     percent_increase = ((train_target_timeseries[i] - start_price) / start_price) * 100
    #     y_train.append(percent_increase)
    #
    # for i in range(0, len(test_target_timeseries)):
    #     start_price = x_test[i][-1][2]
    #     percent_increase = ((test_target_timeseries[i] - start_price) / start_price) * 100
    #     y_test.append(percent_increase)


    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return [x_train, y_train, x_test, y_test, reference_points]



# MAIN()

stock_name = 'GOOGL'
df = get_stock_data(stock_name)

window = 10
target_len = 10
X_train, y_train, X_test, y_test, ref = load_data(df[::-1], window, target_len=target_len, train_percent=.9)

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)


model = build_model([3, window, target_len])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=1,
    validation_split=0.1,
    verbose=2)


trainScore = model.evaluate(X_train, y_train, verbose=100)
print('Train Score: %.2f MSE (%.2f RMSE) (%.2f)' % (trainScore[0], math.sqrt(trainScore[0]), trainScore[1]))

testScore = model.evaluate(X_test, y_test, verbose=100)
print('Test Score: %.2f MSE (%.2f RMSE) (%.2f)' % (testScore[0], math.sqrt(testScore[0]), testScore[1]))

p = model.predict(X_test)


# document results in file
file = open("rnn_output4.txt", "w")
for i in range(0, len(X_train)):
    for s in range(0, window):
        file.write(str(X_train[i][s]) + "\n")
    file.write("Target: " + str(y_train[i]) + "\n")
    file.write("\n")

for i in range(0, len(X_test)):
    for s in range(0, window):
        file.write(str(X_test[i][s]) + "\n")
    file.write("Target: " + str(y_test[i]) + "\n")
    file.write("Prediction: " + str(p[i]) + "\n")
    file.write("\n")


# de-normalize
for i in range(0, len(p)):
    p[i] = (p[i] + 1) * ref[round(.9 * len(ref) + i)]
    y_test[i] = (y_test[i] + 1) * ref[round(.9 * len(ref) + i)]

# plot
plt.plot(p[:, 0], color='red', label='prediction')
plt.plot(y_test[:, 0], color='blue', label='y_test')
plt.legend(loc='upper left')
plt.show()