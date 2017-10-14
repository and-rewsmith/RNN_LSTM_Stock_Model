from urllib.request import urlopen


def read_stocks(url):
    data = urlopen(url)
    tickers = []
    for line in data:
        line = line.decode('utf-8').split("|")
        line = line[0]
        tickers.append(line)
    tickers = tickers[1:-1]
    return tickers