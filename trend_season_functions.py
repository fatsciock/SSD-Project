import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from utility_functions import RMSE
from plot_functions import plot_trend, plot_pearson, plot_seasonal_decomposition, \
    plot_notrend_noseason, plot_model, plot_prediction


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


def run_TREND_SEASON(museum_visitors, number_of_measurements, period_to_predict, extended_dates):
    # Individuazione del trend
    trend, regression_params = fit_trend_model(museum_visitors)

    # Ricerca della stagionalità tramite l'indice di Pearson
    seasonality = find_seasonality(museum_visitors)["index"]

    # Decomposizione dei dati: vengono mostrati il trend, la stagionalità e i residui
    plot_seasonal_decomposition(museum_visitors, seasonality)

    # Eliminazione del trend
    notrend = museum_visitors.Visitors.to_numpy() / trend

    # Ricerca coefficienti di stagionalità
    season_coeff = []
    for i in range(seasonality):
        temp = []
        for j in range(i, number_of_measurements, seasonality):
            temp.append(notrend[j])
        season_coeff.append(np.mean(temp))

    # Partendo dai dati detrendizzati viene eliminata anche la stagionalità
    noseason = []
    for i in range(len(trend)):
        noseason.append(notrend[i] / season_coeff[i % seasonality])

    # Viene ricreata la funzione che modella i dati a partire dal trend e dai coefficienti di stagionalità trovati
    trend_season = []
    for i in range(len(trend)):
        trend_season.append(trend[i] * season_coeff[i % seasonality])

    # Plot dei dati senza trend e senza stagionalità
    plot_notrend_noseason(notrend, noseason)

    # Plot del modello ottenuto
    plot_model(museum_visitors, trend_season)

    # Costruzione del modello tramite il quale prevedere i prossimi 24 periodi.
    # Vengono utilizzati i coefficienti della retta di regressione del trend e
    # i coefficienti di stagionalità trovati precedentemente
    len_to_predict = number_of_measurements + period_to_predict
    x_predict = np.linspace(0, len_to_predict - 1, len_to_predict)
    y_predict = linear_trend(x_predict, regression_params[0], regression_params[1])

    predicted = []
    for i in range(len(x_predict)):
        predicted.append(y_predict[i] * season_coeff[i % seasonality])

    # Plot della previsione effettuata
    plot_prediction(museum_visitors, trend_season, predicted, y_predict, extended_dates)

    # Calcolo dell'errore commesso dal modello
    rmse = RMSE(museum_visitors.Visitors.to_numpy(), np.array(trend_season))
    print("La loss del modello calcolata tramite RMSE è pari a {}\n".format(round(rmse, 3)))
