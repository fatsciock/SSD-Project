import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utility_functions import RMSE, create_dataset


def run_LSTM(museum_visitors, dates):
    museum_visitors.Visitors = museum_visitors.Visitors.astype('float32')
    cutpoint = int(0.8 * len(museum_visitors.Visitors))
    look_back = 12

    # Preprocessing dei dati
    scaler = MinMaxScaler((0.1, 0.9))
    museum_visitors['Scaled'] = scaler.fit_transform(museum_visitors.Visitors.to_numpy().reshape(-1, 1))

    train_set_scaled = museum_visitors.Scaled[:cutpoint]
    test_set_scaled = museum_visitors.Scaled[cutpoint:]

    testdata = np.concatenate((train_set_scaled[-look_back:], test_set_scaled))
    trainX, trainY = create_dataset(train_set_scaled, look_back)
    testX, testY = create_dataset(testdata, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    lstm_model = Sequential()
    n_input = look_back
    n_hidden = 20
    n_output = 1
    lstm_model.add(LSTM(n_hidden, activation="relu", input_shape=(n_output, n_input), dropout=0.05))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(trainX, trainY, epochs=100, batch_size=1)
    print(lstm_model.summary())

    trainPredict = lstm_model.predict(trainX)
    testForecast = lstm_model.predict(testX)

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testForecast = scaler.inverse_transform(testForecast)
    testY = scaler.inverse_transform([testY])

    trainscore = RMSE(trainY[0], trainPredict[:, 0])
    print("RMSE train: {}".format(round(trainscore, 3)))
    testscore = RMSE(testY[0], testForecast[:, 0])
    print("RMSE test: {}".format(round(testscore, 3)))

    plt.title("LSTM")
    plt.plot(museum_visitors.Visitors.to_numpy(), label="Dati originali")
    plt.plot(np.linspace(look_back, cutpoint - 1, cutpoint - look_back), trainPredict, label="Prediction")
    plt.plot(np.linspace(cutpoint, cutpoint + len(testForecast) - 1, len(testForecast)), testForecast, label="Forecast")
    plt.legend()
    plt.show()
