import arrow
import quandl


num_days_back = 100
stock_ticker = "AAL"


print("GETTING STOCK DATA")

end_date = arrow.now().format("YYYY-MM-DD")
start_date = arrow.now()
start_date = start_date.replace(days=(num_days_back * -1)).format("YYYY-MM-DD")

quandl_api_key = "DqEaArDZQP8SfgHTd_Ko"
quandl.ApiConfig.api_key = quandl_api_key

source = "WIKI/" + stock_ticker

data = quandl.get(source, start_date=str(start_date), end_date=str(end_date))

print(len(data))

data = data[["Open", "High", "Low", "Volume", "Close"]].as_matrix()