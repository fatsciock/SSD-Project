import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
import os


def plot_data(museum):
    period_without_covid = 74
    visitors_before_covid = museum.iloc[:period_without_covid]
    visitors_during_covid = museum.iloc[period_without_covid - 1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Grafici dei dati con e senza le misurazioni nel periodo COVID-19")

    ax1.plot(visitors_before_covid.Visitors, label="Visitatori in assenza del COVID-19")
    ax1.plot(visitors_during_covid.Visitors, 'r-.', label="Visitatori durante il periodo COVID-19")
    ax1.legend()

    # Grafico dei valori dei visitatori al museo Avila Adobe che verranno utilizzati da qui in avanti
    ax2.plot(visitors_before_covid.Visitors, label="Visitatori in assenza del COVID-19")
    ax2.legend()

    plt.show()


def plot_pearson(pearson_indexes, max_pearson):
    plt.title("Individuazione della stagionalità tramite l'indice di Pearson")
    x = np.linspace(0, len(pearson_indexes) - 1, len(pearson_indexes))
    barlist = plt.bar(x, pearson_indexes)
    barlist[max_pearson["index"]].set_color("r")
    plt.text(max_pearson["index"], 0.95,
             "Valore massimo: {0}\nIndice: {1}".format(round(max_pearson["pearson"], 2), max_pearson["index"]),
             fontsize=8, color="red", ha="center")
    plt.show()


def plot_seasonal_decomposition(museum, period):
    mul = seasonal_decompose(museum.Visitors, model='multiplicative', period=period)
    add = seasonal_decompose(museum.Visitors, model='additive', period=period)

    plt.semilogy(mul.resid, 'bo', label="Modello moltiplicativo")
    plt.semilogy(add.resid, 'ro', label="Modello additivo")
    plt.yscale("symlog")
    plt.legend()
    plt.plot()
    plt.title("Confronto dei residui tra modello moltiplicativo e additivo")
    plt.show()

    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    mul.plot()
    plt.show()


def plot_trend(data, yfit):
    plt.title("Individuazione del trend")
    plt.plot(data, label="Dati originali")
    plt.plot(yfit, label="Trend lineare decrescente")
    plt.legend()
    plt.show()


def plot_notrend_noseason(nt, ns):
    plt.title("Eliminazione trend e stagionalità")
    plt.plot(nt, label="Dati senza trend")
    plt.plot(ns, label="Dati senza trend e stagionalità")
    plt.legend()
    plt.show()


def plot_model(museum, ts):
    data = museum.Visitors
    plt.title("Comparazione dati originali con il modello ottenuto")
    plt.plot(np.linspace(0, len(data) - 1, len(data)), data, label="Dati originali")
    plt.plot(ts, 'r--', label="Modello")
    plt.legend()
    plt.show()


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
    for i in range(1, shift):
        tmp, _ = pearsonr(data[shift:], data[shift-i: -i])
        pearson_indexes.append(tmp)
        # Si considera un valore superiore a 0.7 come indice di forte correlazione
        if abs(tmp) > max_pearson["pearson"] and abs(tmp) > 0.7:
            max_pearson["index"] = i
            max_pearson["pearson"] = tmp

    plot_pearson(pearson_indexes, max_pearson)

    return max_pearson


def linear_trend(x, m, q):
    return np.array(m * x + q)


def RMSE(y_actual, y_predicted):
    return np.sqrt(np.mean((y_predicted - y_actual) ** 2))


def plot_prediction(museum, trend_season_data, predicted_data, regression, n1, n2):
    data = museum.Visitors

    plt.plot(np.linspace(0, n1 - 1, n1),
             data, label="Dati originali")
    plt.plot(np.linspace(0, n1 - 1, n1),
             trend_season_data, 'r--', label="Modello")
    plt.plot(np.linspace(73, n2, n2 - 73),
             predicted_data[73:], '--', label="Previsione")
    plt.plot(regression, label="Trend")
    plt.title("Previsione dei dati su 24 periodi")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    museum_visitors = pd.read_csv("Data/museum-visitors.csv", parse_dates=["Month"])

    # Si è scelto di lavorare sui visitatori del museo 'Avila Adobe'
    museum_name = "Avila Adobe"
    avila_adobe_visitors = museum_visitors.loc[:, ["Month", museum_name]]

    # Si rinomina la colonna relativa ai visitatori del museo e si imposta quella delle date come indice
    avila_adobe_visitors.rename(columns={museum_name: 'Visitors'}, inplace=True)
    avila_adobe_visitors.set_index('Month', inplace=True)

    # Ora vengono mostrati i dati in un grafico che evidenzia i valori misurati durante la pandemia
    # del COVID-19. Tali dati non verranno usati per tutte le elaborazioni successive.
    # Viene anche mostrato un grafico dei dati esenti dagli effetti della pandemia
    plot_data(avila_adobe_visitors)

    # Esclusione dati del periodo COVID
    avila_adobe_visitors = avila_adobe_visitors.iloc[:74]
    number_of_measurements = len(avila_adobe_visitors.Visitors)

    # Individuazione del trend
    trend, regression_params = fit_trend_model(avila_adobe_visitors)

    # Ricerca della stagionalità tramite l'indice di Pearson
    seasonality = find_seasonality(avila_adobe_visitors)
    number_of_season = seasonality["index"]

    # Decomposizione dei dati: vengono mostrati il trend, la stagionalità e i residui
    plot_seasonal_decomposition(avila_adobe_visitors, number_of_season)

    # Eliminazione del trend
    notrend = avila_adobe_visitors.Visitors.to_numpy() / trend

    # Ricerca coefficienti di stagionalità
    season_coeff = []
    for i in range(number_of_season):
        temp = []
        for j in range(i, number_of_measurements, number_of_season):
            temp.append(notrend[j])
        season_coeff.append(np.mean(temp))

    # Partendo dai dati detrendizzati viene eliminata anche la stagionalità
    noseason = []
    for i in range(len(trend)):
        noseason.append(notrend[i] / season_coeff[i % number_of_season])

    # Viene ricreata la funzione che modella i dati a partire dal trend e dai coefficienti di stagionalità trovati
    trend_season = []
    for i in range(len(trend)):
        trend_season.append(trend[i] * season_coeff[i % number_of_season])

    # Plot dei dati senza trend e senza stagionalità
    plot_notrend_noseason(notrend, noseason)

    # Plot del modello ottenuto
    plot_model(avila_adobe_visitors, trend_season)

    # Costruzione del modello tramite il quale prevedere i prossimi 24 periodi.
    # Vengono utilizzati i coefficienti della retta di regressione del trend e
    # i coefficienti di stagionalità trovati precedentemente
    period_to_predict = 24
    len_to_predict = number_of_measurements + period_to_predict
    x_predict = np.linspace(0, len_to_predict - 1, len_to_predict)
    y_predict = linear_trend(x_predict, regression_params[0], regression_params[1])

    predicted = []
    for i in range(len(x_predict)):
        predicted.append(y_predict[i] * season_coeff[i % number_of_season])

    # Plot della previsione effettuata
    plot_prediction(avila_adobe_visitors, trend_season, predicted, y_predict, number_of_measurements, len_to_predict)

    # Calcolo dell'errore commesso dal modello
    rmse = RMSE(avila_adobe_visitors.Visitors.to_numpy(), np.array(trend_season))
    print("La loss del modello calcolata tramite RMSE è pari a {}".format(round(rmse, 3)))

'''
PARTE 1
- Trovare funzione di trend e parameter fitting - DONE
- Eliminare trend - DONE
- Determinare se utilizzare un modello moltiplicativo o additivo - DONE
- Individuare stagionalità - DONE
- Destagionalizzare - DONE
- Fare previsione con funzione di trend - DONE  

PARTE 2
- Fare previsione con modelli predittivi statistici e neurali
- Analizzare e confrontare i risultati
'''
