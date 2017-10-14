import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Version1__base.model_mult_tar import build_model
import arrow
import quandl
from send_email import send_email
from read_tickers import read_stocks
np.set_printoptions(suppress=True)



#gets stock data from quandl in the form of a np array
def get_stock_data(stock_ticker, num_days_back, minimum_days):
    print("GETTING STOCK DATA")

    end_date = arrow.now().format("YYYY-MM-DD")
    start_date = arrow.now()
    start_date = start_date.replace(days=(num_days_back*-1)).format("YYYY-MM-DD")

    quandl_api_key = "DqEaArDZQP8SfgHTd_Ko"
    quandl.ApiConfig.api_key = quandl_api_key

    source = "WIKI/" + stock_ticker

    print("    Retrieving data from quandl API...")
    data = quandl.get(source, start_date=str(start_date), end_date=str(end_date))
    data = data[["Open", "High", "Low", "Volume", "Close"]].as_matrix()

    if len(data) < minimum_days:
        raise quandl.errors.quandl_error.NotFoundError

    return data



def normalize_timestep(timestep, reference_list):
    reference_price = timestep[0][0]
    reference_list.append(reference_price)

    temp_volume = np.copy(timestep[:, 3])
    reference_volume = np.copy(timestep[0, 3])

    timestep = (timestep / reference_price) - 1
    timestep[:, 3] = (temp_volume / reference_volume) - 1
    return timestep



#take data and split into timeseries so that we can train the model
def load_data(stock_data, num_timesteps, target_len, train_percent=.75):

    # iterate so that we can also capture a sequence for a target
    combined_length = num_timesteps + target_len

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
        result[i] = normalize_timestep(result[i], reference_points)


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



def generate_graph(stock_name, days_back, num_timesteps, target_len, minimum_days=500):
    stock_name = stock_name
    stock_data = get_stock_data(stock_name, days_back, minimum_days)

    X_train, y_train, X_test, y_test, ref = load_data(stock_data, num_timesteps, target_len=target_len, train_percent=.9)

    # store recent data so that we can get a live prediction
    recent_reference = []
    recent_data = stock_data[-num_timesteps:]
    recent_data = normalize_timestep(recent_data, recent_reference)

    print("    X_train", X_train.shape)
    print("    y_train", y_train.shape)
    print("    X_test", X_test.shape)
    print("    y_test", y_test.shape)

    # setup model
    print("TRAINING")
    model = build_model([5, num_timesteps, target_len])
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        epochs=1,
        validation_split=0.1,
        verbose=2)

    #train the model
    trainScore = model.evaluate(X_train, y_train, verbose=100)
    print('Train Score: %.2f MSE (%.2f RMSE) (%.2f)' % (trainScore[0], math.sqrt(trainScore[0]), trainScore[1]))

    testScore = model.evaluate(X_test, y_test, verbose=100)
    print('Test Score: %.2f MSE (%.2f RMSE) (%.2f)' % (testScore[0], math.sqrt(testScore[0]), testScore[1]))

    #make predictions
    print("PREDICTING")
    p = model.predict(X_test)
    recent_data = [recent_data] # One-sample predictions need list wrapper. Argument must be 3d.
    recent_data = np.asarray(recent_data)
    future = model.predict([recent_data])

    # document results in file
    print("WRITING TO LOG")
    file = open("log.txt", "w")
    for i in range(0, len(X_train)):
        for s in range(0, num_timesteps):
            file.write(str(X_train[i][s]) + "\n")
        file.write("Target: " + str(y_train[i]) + "\n")
        file.write("\n")

    for i in range(0, len(X_test)):
        for s in range(0, num_timesteps):
            file.write(str(X_test[i][s]) + "\n")
        file.write("Target: " + str(y_test[i]) + "\n")
        file.write("Prediction: " + str(p[i]) + "\n")
        file.write("\n")

    # de-normalize
    print("DENORMALIZING")
    for i in range(0, len(p)):
        p[i] = (p[i] + 1) * ref[round(.9 * len(ref) + i)]
        y_test[i] = (y_test[i] + 1) * ref[round(.9 * len(ref) + i)]

    future[0] = (future[0] + 1) * recent_reference[0]
    recent_data[0] = (recent_data[0] + 1) * recent_reference[0]

    # plot historical predictions
    print("PLOTTING")
    for i in range(0, len(p)):
        if i % (target_len*2) == 0:
            plot_index = i #for filling plot indexes
            plot_indexes = []
            plot_values = p[i]
            for j in range(0, target_len):
                plot_indexes.append(plot_index)
                plot_index += 1
            plt.plot(plot_indexes, plot_values, color="red")

    # plot historical actual
    plt.plot(y_test[:, 0], color='blue', label='Actual') # actual stock price history

    # plot recent prices
    plot_indexes = [len(y_test) - 1]
    plot_values = [y_test[-1, 0]]
    plot_index = None
    for i in range(0, len(recent_data[0])):
        plot_values.append(recent_data[0][i][0])
        plot_index = len(y_test) + i
        plot_indexes.append(len(y_test)+i)
    plt.plot(plot_indexes, plot_values, color='blue')

    # plot future predictions
    plot_indexes = [plot_index]
    plot_values = [recent_data[0][-1][0]]
    for i in range(0, len(future[0])):
        plot_index += 1
        plot_values.append(future[0][i])
        plot_indexes.append(plot_index)
    plt.plot(plot_indexes, plot_values, color="red", label="Prediction")

    #show/save plot
    print("SENDING EMAILS")
    plt.legend(loc="upper left")
    plt.title(stock_name + " Price Predictions")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    filename = stock_name + "_" + str(arrow.utcnow().format("YYYY-MM-DD") + "_" + str(days_back))
    plt.savefig("graphs/" + filename)
    #plt.show()
    plt.close()
    send_email(filename)

    return True



# MAIN()

tickers = read_stocks("ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt")
num_days_back = 3700

for ticker in tickers:
    print("Ticker:" + str(ticker))

    try:
        isDone = generate_graph(ticker, num_days_back, 100, 30)
    except quandl.errors.quandl_error.NotFoundError:
        continue

    # generate_graph(ticker, 300, 20, 10) #FOR TESTING
