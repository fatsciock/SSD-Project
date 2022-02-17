import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from plot_functions import plot_MLP_forecasts
from utility_functions import RMSE, create_dataset


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


def run_MLP(museum_visitors, dates, periods_to_forecast):
    print("---------------MLP---------------")
    museum_visitors.Visitors = museum_visitors.Visitors.astype('float32')
    cutpoint = int(0.8 * len(museum_visitors.Visitors))

    # Preprocessing dei dati
    scaler = MinMaxScaler((0.1, 0.9))
    scaler.fit(museum_visitors.Visitors.to_numpy().reshape(-1, 1))
    museum_visitors['Scaled'] = scaler.transform(museum_visitors.Visitors.to_numpy().reshape(-1, 1))

    train_set_scaled = museum_visitors.Scaled[:cutpoint]
    test_set_scaled = museum_visitors.Scaled[cutpoint:]

    look_back = 12

    data_for_testing_scaled = np.concatenate([train_set_scaled[-look_back:], test_set_scaled])

    training_set_x_scaled, training_set_y_scaled = create_dataset(train_set_scaled, look_back)
    test_set_x_scaled, test_set_y_scaled = create_dataset(data_for_testing_scaled, look_back)

    # Costruzione rete neurale
    neural_net = Sequential()
    # strato di input e primo livello nascosto
    neural_net.add(Dense(look_back * 2, input_dim=look_back, activation='relu'))
    # Livello di output a singolo neurone
    neural_net.add(Dense(1))

    neural_net.compile(loss='mean_squared_error', optimizer='adam')
    neural_net.fit(training_set_x_scaled, training_set_y_scaled,
                   batch_size=2, epochs=100,
                   verbose=0, workers=-1, use_multiprocessing=True)

    # Previsioni sul training e sul test set
    predictions_train = neural_net.predict(training_set_x_scaled)
    predictions_test = neural_net.predict(test_set_x_scaled)
    forecasts = forecast_visitors(neural_net, museum_visitors.Scaled, periods_to_forecast)

    # Riscalo i dati nel formato originale
    predictions_train = scaler.inverse_transform(predictions_train)
    trainY = scaler.inverse_transform([training_set_y_scaled])
    predictions_test = scaler.inverse_transform(predictions_test)
    testY = scaler.inverse_transform([test_set_y_scaled])
    forecasts = scaler.inverse_transform(forecasts)

    print("La loss del modello MLP è:")
    trainscore = RMSE(trainY[0], predictions_train[:, 0])
    print('RMSE train: {}'.format(round(trainscore, 3)))
    testscore = RMSE(testY[0], predictions_test[:, 0])
    print('RMSE test: {}'.format(round(testscore, 3)))

    plot_MLP_forecasts(museum_visitors, predictions_train, predictions_test, forecasts, look_back, cutpoint, dates)

    return np.concatenate((predictions_train, predictions_test)).reshape((-1, ))
