import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from utility_functions import RMSE, create_dataset
from plot_functions import plot_LSTM_forecasts


def forecast_visitors(neural_net, museum_visitors, periods_to_forecast, look_back):
    values_for_forecast = museum_visitors.Scaled[-look_back:].to_numpy().reshape((-1, 1, look_back))
    forecasts = []

    for step in range(periods_to_forecast):
        next_forecast = neural_net.predict(values_for_forecast)[0]
        values_for_forecast[0][0][0] = next_forecast
        values_for_forecast = np.reshape(np.roll(values_for_forecast[0][0], -1), (-1, 1, look_back))
        forecasts.append(next_forecast)

    return forecasts


def run_LSTM(museum_visitors, dates, periods_to_forecast, cutpoint):
    museum_visitors.Visitors = museum_visitors.Visitors.astype('float32')

    look_back = 12
    n_input = look_back
    n_hidden = 20
    n_output = 1

    # Preprocessing dei dati
    scaler = MinMaxScaler((0.1, 0.9))
    museum_visitors['Scaled'] = scaler.fit_transform(museum_visitors.Visitors.to_numpy().reshape(-1, 1))

    train_scaled = museum_visitors.Scaled[:cutpoint]
    test_scaled = museum_visitors.Scaled[cutpoint:]
    test_scaled = np.concatenate((train_scaled[-look_back:], test_scaled))

    trainX, trainY = create_dataset(train_scaled, look_back)
    testX, testY = create_dataset(test_scaled, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Costruzione NN
    lstm_model = Sequential()
    # Aggiunta del input e hidden layer
    lstm_model.add(LSTM(n_hidden, activation="relu", input_shape=(n_output, n_input), dropout=0.05))
    # Aggiunta output layer
    lstm_model.add(Dense(1))

    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=0)
    # print(lstm_model.summary())

    # Previsioni sul training set, test set, e per 24 periodi futuri
    train_predict = lstm_model.predict(trainX)
    test_predict = lstm_model.predict(testX)
    forecasts = forecast_visitors(lstm_model, museum_visitors, periods_to_forecast, look_back)

    # Scaling dei dati nel formato originale
    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform([trainY])
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform([testY])
    forecasts = scaler.inverse_transform(forecasts)

    print("\n---------------LSTM---------------")
    print("La loss del modello LSTM è:")
    trainscore = RMSE(trainY[0], train_predict[:, 0])
    print("RMSE train: {}".format(round(trainscore, 3)))
    testscore = RMSE(testY[0], test_predict[:, 0])
    print("RMSE test: {}".format(round(testscore, 3)))

    plot_LSTM_forecasts(museum_visitors, train_predict, test_predict, forecasts, look_back, cutpoint, dates)

    return test_predict.reshape(-1, )
