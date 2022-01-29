import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from utility_functions import RMSE
from plot_functions import plot_trend, plot_pearson


def linear_trend(x, m, q):
    return np.array(m * x + q)


def fit_trend_model(museum):
    x = np.linspace(0, len(museum) - 1, len(museum))
    initial_guess = [-1, 1]
    data = museum.Visitors.to_numpy()

    best_params, _ = curve_fit(linear_trend, x, data, p0=initial_guess)
    print("I migliori parametri per la retta di regressione sono: m={0}, q={1}".format(round(best_params[0], 2),
                                                                                       round(best_params[1], 2)))
    yfit = linear_trend(x, best_params[0], best_params[1])
    rmse = RMSE(data, yfit)
    print("La loss del trend calcolata tramite RMSE è pari a {}".format(round(rmse, 3)))

    plot_trend(data, yfit)

    return yfit, best_params


def find_seasonality(museum):
    data = museum.Visitors.to_numpy()
    max_pearson = {"index": 0, "pearson": 0}

    # Il primo elemento si otterrebbe con pearsonr(data[24:], data[24:]), cioè confrontando tra loro
    # gli stessi identici dati.
    pearson_indexes = [1]

    shift = 24
    for idx in range(1, shift):
        tmp, _ = pearsonr(data[shift:], data[shift-idx: -idx])
        pearson_indexes.append(tmp)
        # Si considera un valore superiore a 0.7 come indice di forte correlazione
        if abs(tmp) > max_pearson["pearson"] and abs(tmp) > 0.7:
            max_pearson["index"] = idx
            max_pearson["pearson"] = tmp

    plot_pearson(pearson_indexes, max_pearson)

    return max_pearson
