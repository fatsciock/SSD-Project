import numpy as np
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from plot_functions import plot_SARIMA_predicition, plot_diagnostic
from utility_functions import print_loss


def run_statistical_algorithm(number_of_measurements, museum_visitors):
    cutpoint = int(0.7 * number_of_measurements)
    train_set = museum_visitors.Visitors[:cutpoint]
    test_set = museum_visitors.Visitors[cutpoint:]
    period_to_predict = 24
    x = np.linspace(0, number_of_measurements - 1, number_of_measurements)
    xfore = np.linspace(number_of_measurements, number_of_measurements - 1 + period_to_predict, period_to_predict)

    auto_m = pm.auto_arima(train_set, start_p=0, max_p=12, start_q=0, max_q=12,
                           start_P=0, max_P=2, start_Q=0, max_Q=2, m=12, stepwise=False,
                           seasonal=True, trace=False, error_action="ignore", suppress_warnings=True)

    print("Valori trovati con auto_arima:\n"
          "(p,d,q): {0}\n"
          "(P,D,Q,m): {1}".format(auto_m.order, auto_m.seasonal_order))

    sarima_m = SARIMAX(train_set, order=auto_m.order, seasonal_order=auto_m.seasonal_order)
    sarima_fit_m = sarima_m.fit(disp=False, maxiter=100)

    # plot_diagnostic(sarima_fit_m)

    ypred = sarima_fit_m.predict(start=1, end=len(train_set))
    forecast = sarima_fit_m.get_forecast(steps=len(test_set) + period_to_predict)
    yfore = forecast.predicted_mean

    plot_SARIMA_predicition(x, xfore, yfore, ypred, forecast.conf_int(), cutpoint, period_to_predict, museum_visitors)

    print_loss(yfore[:-period_to_predict], ypred, museum_visitors.Visitors.to_numpy(), cutpoint)
