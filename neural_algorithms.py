# PER MLP RICORDARSI DI SCALARE I VALORI DI INPUT IN INTERVALLO PIù PICCOLO DEL MINIMO E MASSIMO
# AD ESEMPIO TRA [0.1, 0.8]

import math
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def run_neural_algorithms(museum_visitors, seasonality):
    museum_visitors.Visitors = museum_visitors.Visitors.astype('float32')
    cutpoint = int(0.8 * len(museum_visitors.Visitors))
    train_set = museum_visitors.Visitors[:cutpoint]
    test_set = museum_visitors.Visitors[cutpoint:]

    period_to_predict = 24
    look_back = seasonality

    data_for_testing = np.concatenate([train_set[-look_back:].Visitors, test_set.Visitors])

    training_set_x, training_set_y = create_dataset(train_set.Visitors, look_back)
    test_set_x, test_set_y = create_dataset(test_set.Visitors, look_back)

    neural_net = Sequential()
    # strato di input e primo livello nascosto
    neural_net.add(Dense(look_back*2+1, input_dim=look_back, activation='relu'))
    # Livello di output a singolo neurone
    neural_net.add(Dense(1))

    neural_net.compile(loss='mean_squared_error', optimizer='adam')
    neural_net.fit(training_set_x, training_set_y,
                   batch_size=4, epochs=200,
                   verbose=2, workers=-1, use_multiprocessing=True)




