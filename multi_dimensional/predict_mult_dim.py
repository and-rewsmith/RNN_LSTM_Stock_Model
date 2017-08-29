import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiple__targets.model_mult_tar import build_model


#gets stock data from google finance, excludes some columns, then returns a dataframe
def get_stock_data(stock_name):
    print("GETTING STOCK DATA")
    url = "http://www.google.com/finance/historical?q=" + stock_name + "&startdate=Jul+10%2C+2005&enddate=Aug+20%2C+2017&num=30&ei=rCtlWZGSFN3KsQHwrqWQCw&output=csv"

    col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    stocks = pd.read_csv(url, header=0, names=col_names)
    df = pd.DataFrame(stocks)
    df.drop(df.columns[[0, 3, 5]], axis=1, inplace=True)

    return df