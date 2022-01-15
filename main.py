import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime as dt
import os


def difference(data, interval):
    return np.array([data[i] - data[i - interval] for i in range(interval, len(data))])


def invert_difference(original_data, diff_data, interval):
    return np.array(
        [diff_data[i - interval] + original_data[i - interval] for i in range(interval, len(original_data))])


def normalize(data):
    return (data - min(data)) / (max(data) - min(data))


def log_norm(x, mu, sigma):
    return 1 / (x * np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))


# def parameter_fitting(data):
#     x = np.linspace(1, 74, 74)
#     true_val = [1, 1]
#     best_vals, _ = curve_fit(log_norm, x, data, p0=true_val)
#     f = log_norm(x, best_vals[0], best_vals[1])
#     plt.plot(f)
#     plt.plot(data)
#     plt.show()

def model(x, m, q):
    return m * x + q


def loss(ytrue, ymod):
    return np.sqrt(np.mean((ymod - ytrue) ** 2))


def parameter_fitting(data):
    x = np.linspace(1, 74, 74)
    initial_guess = [-1, 1]
    vals, _ = curve_fit(model, x, data, p0=initial_guess)
    f = model(x, vals[0], vals[1])
    print('CF loss: {}'.format(loss(data, f)))
    plt.plot(x, f)
    plt.plot(x, data)
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    museum_visitors = pd.read_csv("../../csv/museum-visitors.csv")

    # Si è scelto di lavorare sui visitatori del museo 'Avila Adobe'
    museum_name = "Avila Adobe"
    avila_adobe_visitors = museum_visitors.loc[:, ["Month", museum_name]]

    # Qui si è modificato il formato della data, in modo da renderlo meglio leggibile
    avila_adobe_visitors['Month'] = pd.to_datetime(avila_adobe_visitors["Month"],
                                                   format="%Y-%d-%mT%H:%M:%S.%f").dt.strftime('%d/%m/%Y')

    # Infine si è impostata la colonna contenente le date come indice del DataFrame
    avila_adobe_visitors.set_index('Month', inplace=True)

# plt.plot(visitors, label="original")
# plt.legend()
# plt.show()
#
# diff = difference(visitors, 1)
# plt.plot(diff, label="diff")
# plt.legend()
# plt.show()
#
# log = np.log(visitors)
# plt.plot(log, label="log")
# plt.legend()
# plt.show()
#
# logdiff = difference(log, 1)
# plt.plot(logdiff, label="logdiff")
# plt.legend()
# plt.show()

'''

PARTE 1
- Trovare funzione di trend e parameter fitting
- Eliminare trend e determinare se utilizzare un modello moltiplicativo o additivo
- Individuare stagionalità e destagionalizzare
- Fare previsione con funzione di trend

PARTE 2
- Fare previsione con modelli predittivi statistici e neurali
- Analizzare e confrontare i risultati

'''
