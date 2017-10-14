#TODO: FINISH MAIN()

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from Version3__multiple_stocls(Under Development).model_mult_dim import build_model
import arrow
import quandl
np.set_printoptions(suppress=True)



#gets stock data from quandl in the form of a np array
def get_stock_data(main_ticker, helper_tickers, num_days_back):
    print("GETTING STOCK DATA")

    helper_tickers.extend(main_ticker)

    end_date = arrow.now().format("YYYY-MM-DD")
    start_date = arrow.now()
    start_date = start_date.replace(days=(num_days_back*-1)).format("YYYY-MM-DD")

    quandl_api_key = "DqEaArDZQP8SfgHTd_Ko"
    quandl.ApiConfig.api_key = quandl_api_key

    all_features = []
    all_targets = []

    for i in range(0, len(helper_tickers)):
        source = "WIKI/" + helper_tickers[i]
        data = quandl.get(source, start_date=str(start_date), end_date=str(end_date))
        features = data[["Open", "High", "Low", "Volume"]].as_matrix()
        targets = data[["Close"]].as_matrix()
        all_features.append(features)
        all_targets.append(targets)

    all_features = np.asarray(all_features)
    all_targets = np.asarray(all_targets)

    all_features = np.hstack((all_features))
    all_targets = np.hstack((all_targets))
    all_features = np.hstack((all_features, all_targets))

    return all_features



#take data and split into timeseries so that we can train the model
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
        index = 0
        num_stocks = len(result[0][0]) / 5 #this 5 needs to change if we change the number of features
        for stock_num in range(0, num_stocks):
            if stock_num + 1 == num_stocks:
                reference_points.append(result[i,0,index])
            close_index = len(result[i,0,0])-num_stocks+stock_num
            result[i,:,index:index+3] = result[i,:,index:index+3] / result[i,0,index]
            result[i,:,index+3] = result[i,:,index+3] / result[i,0,index+3]
            result[i,:,close_index] = result[i,:,close_index] / result[i,0,index]
            index += 4

    # train test split
    row = round(train_percent * result.shape[0])
    train = result[:int(row), :]
    test = result[int(row):, :]

    split_index = len(train[0]) - target_len
    x_train = train[:, :split_index]
    y_train = train[:, split_index:, -1]

    x_test = test[:, :split_index]
    y_test = test[:, split_index:, -1]

    return [x_train, y_train, x_test, y_test, reference_points]






# MAIN()

main_stock = ['GOOGL']
helper_stocks = ["AAPL"]

stock_data = get_stock_data(main_stock, helper_stocks, 300)

window = 10
target_len = 5
X_train, y_train, X_test, y_test, ref = load_data(stock_data, window, target_len=target_len, train_percent=.9)



