import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Version_1_base.model_mult_tar import build_model
import arrow
import quandl

np.set_printoptions(suppress=True)


#gets stock data from google finance, excludes some columns, then returns a dataframe
def get_stock_data(stock_ticker, num_days_back):
    print("GETTING STOCK DATA")

    end_date = arrow.now().format("YYYY-MM-DD")
    start_date = arrow.now()
    start_date = start_date.replace(days=(num_days_back*-1)).format("YYYY-MM-DD")

    quandl_api_key = "DqEaArDZQP8SfgHTd_Ko"
    quandl.ApiConfig.api_key = quandl_api_key

    source = "WIKI/" + stock_ticker

    data = quandl.get(source, start_date=str(start_date), end_date=str(end_date))
    data = data[["Open", "High", "Low", "Volume", "Close"]].as_matrix()
    return data




#Loads in stock data from a dataframe and tra
def load_data(stock_data, seq_len, target_len, train_percent=.75):

    # iterate so that we can also capture a sequence for a target
    combined_length = seq_len + target_len

    print("SPLITTING INTO TIMESERIES")

    # segment the data into timeseries (these will be overlapping)
    result = []
    for index in range(len(stock_data) - combined_length):
        time_series = stock_data[index: index + combined_length]
        result.append(time_series[:])

    result = np.asarray(result)

    # normalize
    reference_points = [] #for de-normalizing outside of the function
    for i in range(0, len(result)):
        #store ref point for graphing later
        reference_points.append(result[i][0][0])

        #temporary information for normalizing volume
        temp_volume = np.copy(result[i, :, 3])
        reference_volume = np.copy(result[i, 0, 3])

        #normalize the prices
        result[i] = (result[i] / result[i][0][0]) - 1
        #normalize volumes
        result[i,:,3] = temp_volume
        result[i,:,3] = (result[i,:,3] / reference_volume) - 1


    # train test split
    row = round(train_percent * result.shape[0])
    train = result[:int(row), :]
    test = result[int(row):, :]

    split_index = len(train[0]) - target_len
    x_train = train[:, :split_index]
    y_train = train[:, split_index:, -1]

    x_test = test[:, :split_index]
    y_test = test[:, split_index:, -1]


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
df = get_stock_data(stock_name, 300)

window = 200
target_len = 50
X_train, y_train, X_test, y_test, ref = load_data(df[::-1], window, target_len=target_len, train_percent=.9)

print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)


model = build_model([5, window, target_len])

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
for i in range(0, len(p)):
    if i % (target_len*2) == 0:
        plot_index = i #for filling plot indexes
        plot_indexes = []
        plot_values = p[i]
        for j in range(0, target_len):
            plot_indexes.append(plot_index)
            plot_index += 1
        plt.plot(plot_indexes, plot_values)

#plt.plot(p[0], color='red', label='prediction') # for single target
plt.plot(y_test[:, 0], color='blue', label='y_test') # actual stock price history
plt.legend(loc='upper left')
plt.show()