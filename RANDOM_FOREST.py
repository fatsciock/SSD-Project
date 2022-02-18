import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from utility_functions import RMSE, create_dataset
from plot_functions import plot_RF_forecasts


def forecast_visitors(neural_net, museum_visitors, periods_to_forecast, look_back):
    values_for_forecast = museum_visitors[-look_back:].reshape((1, look_back))
    forecasts = []

    for step in range(periods_to_forecast):
        next_forecast = neural_net.predict(values_for_forecast)
        values_for_forecast[0][0] = next_forecast
        values_for_forecast = np.reshape(np.roll(values_for_forecast[0], -1), (1, look_back))
        forecasts.append(next_forecast)

    return forecasts


def run_RANDOM_FOREST(museum_visitors, dates, periods_to_forecast, cutpoint):
    museum_visitors.Visitors = museum_visitors.Visitors.astype('float32')

    look_back = 12

    scaler = MinMaxScaler((0.1, 0.9))
    tmp = scaler.fit_transform(museum_visitors.Visitors.to_numpy().reshape(-1, 1))

    train_set = tmp[:cutpoint].reshape(-1,)
    test_set = tmp[cutpoint:].reshape(-1,)
    test_set2 = np.concatenate([train_set[-look_back:], test_set])

    train_set_x, train_set_y = create_dataset(train_set, look_back)
    test_set_x, test_set_y = create_dataset(test_set2, look_back)

    model = RandomForestRegressor(n_estimators=200)
    model.fit(train_set_x, train_set_y)

    train_predict = scaler.inverse_transform(model.predict(train_set_x).reshape(-1, 1))
    test_predict = scaler.inverse_transform(model.predict(test_set_x).reshape(-1, 1))
    forecasts = scaler.inverse_transform(forecast_visitors(model, tmp, periods_to_forecast, look_back))

    print("\n---------------RANDOM FOREST---------------")
    print("La loss del modello Random Forest è:")
    trainscore = RMSE(museum_visitors.Visitors[12:cutpoint].to_numpy(), train_predict.reshape(-1))
    print("RMSE train: {}".format(round(trainscore, 3)))
    testscore = RMSE(museum_visitors.Visitors[cutpoint:].to_numpy(), test_predict.reshape(-1))
    print("RMSE test: {}".format(round(testscore, 3)))

    plot_RF_forecasts(museum_visitors, train_predict, test_predict, forecasts, look_back, cutpoint, dates)

    return test_predict.reshape(-1,)
