# PER MLP RICORDARSI DI SCALARE I VALORI DI INPUT IN INTERVALLO PIù PICCOLO DEL MINIMO E MASSIMO
# AD ESEMPIO TRA [0.1, 0.8]

import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from plot_functions import plot_MLP_forecasts


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def forecast_visitors(neural_net, museum_visitors, periods_to_forecast):
    # Forecast dei visitatori su 24 ulteriori periodi
    values_for_forecast = museum_visitors[-12:].to_numpy().reshape((1, 12))
    forecasts = []

    for step in range(periods_to_forecast):
        next_forecast = neural_net.predict(values_for_forecast)[0]
        values_for_forecast = np.append(values_for_forecast[:, 1:],
                                        np.array(next_forecast).reshape((1, 1)),
                                        axis=1)
        forecasts.append(next_forecast)

    return forecasts


def run_neural_algorithms(museum_visitors, seasonality, dates):
    museum_visitors.Visitors = museum_visitors.Visitors.astype('float32')
    cutpoint = int(0.8 * len(museum_visitors.Visitors))
    train_set = museum_visitors.Visitors[:cutpoint]
    test_set = museum_visitors.Visitors[cutpoint:]

    periods_to_forecast = 24
    look_back = 12

    data_for_testing = np.concatenate([train_set[-look_back:], test_set])

    training_set_x, training_set_y = create_dataset(train_set, look_back)
    test_set_x, test_set_y = create_dataset(data_for_testing, look_back)

    neural_net = Sequential()
    # strato di input e primo livello nascosto
    neural_net.add(Dense(look_back, input_dim=look_back, activation='relu'))
    # neural_net.add(Dense(look_back, activation='relu'))
    # Livello di output a singolo neurone
    neural_net.add(Dense(1))

    neural_net.compile(loss='mean_squared_error', optimizer='adam')
    neural_net.fit(training_set_x, training_set_y,
                   batch_size=4, epochs=300,
                   verbose=2, workers=-1, use_multiprocessing=True)

    trainScore = neural_net.evaluate(training_set_x, training_set_y, verbose=0)
    print('\nTrain Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore, math.sqrt(trainScore)))

    testScore = neural_net.evaluate(test_set_x, test_set_y, verbose=0)
    print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))

    # <Previsioni sul training e sul test set
    predictions_train = neural_net.predict(training_set_x)
    predictions_test = neural_net.predict(test_set_x)

    forecasts = forecast_visitors(neural_net, museum_visitors, periods_to_forecast)

    plot_MLP_forecasts(museum_visitors, predictions_train, predictions_test, forecasts, cutpoint, dates)
