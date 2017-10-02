import quandl
import arrow
import numpy as np

np.set_printoptions(suppress=True)

end_date = arrow.now().format("YYYY-MM-DD")
start_date = arrow.now()
start_date = start_date.replace(days=-20).format("YYYY-MM-DD")

print(start_date)
print(end_date)

quandl_api_key = "DqEaArDZQP8SfgHTd_Ko"
quandl.ApiConfig.api_key = quandl_api_key

data = quandl.get("WIKI/AAPL", start_date=str(start_date), end_date=str(end_date))
format_data = data[["Open", "High", "Low", "Volume", "Close"]]

print(format_data)
print(format_data.as_matrix())
