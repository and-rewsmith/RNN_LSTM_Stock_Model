import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiple__targets.model_mult_tar import build_model
import arrow
import quandl

np.set_printoptions(suppress=True)


#gets stock data from google finance, excludes some columns, then returns a dataframe
def get_stock_data(stock_tickers, num_days_back):
    print("GETTING STOCK DATA")

    end_date = arrow.now().format("YYYY-MM-DD")
    start_date = arrow.now()
    start_date = start_date.replace(days=(num_days_back*-1)).format("YYYY-MM-DD")

    quandl_api_key = "DqEaArDZQP8SfgHTd_Ko"
    quandl.ApiConfig.api_key = quandl_api_key

    all_stock_data = []

    for i in range(0, len(stock_tickers)):
        source = "WIKI/" + stock_tickers[i]
        data = quandl.get(source, start_date=str(start_date), end_date=str(end_date))
        data = data[["Open", "High", "Low", "Volume", "Close"]].as_matrix()
        all_stock_data.append(data)

    return all_stock_data

