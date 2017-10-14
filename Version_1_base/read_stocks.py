from urllib.request import urlopen


def read_stocks(url):
    data = urlopen(url)
    tickers = []
    start = False
    for line in data:
        line = line.decode('utf-8').split("|")
        if start == True:
            line = line[0]
            tickers.append(line)
        if line[0] == "BGFV":
            start = True
    tickers = tickers[1:-1]
    return tickers
