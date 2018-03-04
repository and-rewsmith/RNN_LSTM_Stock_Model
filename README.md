# Recurrent Neural Network (LSTM) Stock Model

## Description

This model uses stock data and twitter sentiment to generate a prediction of future market trends. It gathers sentiment from scraping the Twitter website. Unfortunately, one cannot use the Twitter API as tweets are only available in a window spanning back two weeks. 

General process is as follows:

1. Get stock data from Quandl (day-to-day data)

2. Get stock twitter sentiment for each day

3. Join data

4. Feed it into a Long-term Short-term (LSTM) Neural Network

5. Graph past/future predictions

## Directory Structure

**Version1__base**: Contains a basic implementation. No twitter sentiment analysis.

**Version2__twitter_sentiment**: Contains full functionality. Makes predictions with sentiment analysis included.

**Version3__multiple_stocks(Under Development)**: Work in progress. I plan on this being Version2 with multiple stock pricing data per predictions (for a single stock).

**got/got3**: Get Old Tweets. This repository can be found [here](https://github.com/Jefferson-Henrique/GetOldTweets-python).

**img**: Contains demo graphs.

**misc**: For implementations I might use later. For brainstorming.

**model.py**: The Recurrent Neural Network I am using to train and make predictions.

**read_tickers.py**: Python script to get all of the tickers in NASDAQ.

**scratch.py**: Playground script.

## Performance Examples
![Example_performance_7](https://github.com/als5ev/RNN_LSTM_Stock_Model/blob/master/img/demo7.png?raw=true "Example 7")

![Example_performance_8](https://github.com/als5ev/RNN_LSTM_Stock_Model/blob/master/img/demo8.png?raw=true "Example 8")

![Example_performance_3](https://github.com/als5ev/RNN_LSTM_Stock_Model/blob/master/img/demo3.png?raw=true "Example 3")

![Example performance 4](https://github.com/als5ev/RNN_LSTM_Stock_Model/blob/master/img/demo4.png?raw=true "Example 4")

![Example performance 5](https://github.com/als5ev/RNN_LSTM_Stock_Model/blob/master/img/demo5.png?raw=true "Example 5")

![Example performance 6](https://github.com/als5ev/RNN_LSTM_Stock_Model/blob/master/img/demo6.png?raw=true "Example 6")

![Example performance 2](https://github.com/als5ev/RNN_LSTM_Stock_Model/blob/master/img/demo2.png?raw=true "Example 2")

![Example performance 1](https://github.com/als5ev/RNN_LSTM_Stock_Model/blob/master/img/demo1.png?raw=true "Example 1")


