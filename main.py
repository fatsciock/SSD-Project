import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
import os


def plot_data(data):
    visitors_before_covid = data.iloc[:74]
    visitors_during_covid = data.iloc[73:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Grafici dei dati con e senza le misurazioni nel periodo COVID-19")

    ax1.plot(visitors_before_covid.Visitors, label="Visitatori in assenza del COVID-19")
    ax1.plot(visitors_during_covid.Visitors, 'r-.', label="Visitatori durante il periodo COVID-19")
    ax1.legend()

    # Grafico dei valori dei visitatori al museo Avila Adobe che verranno utilizzati da qui in avanti
    ax2.plot(visitors_before_covid.Visitors, label="Visitatori in assenza del COVID-19")
    ax2.legend()

    plt.show()


def plot_seasonal_decomposition(museum, found_period):
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    result = seasonal_decompose(museum.Visitors, model='multiplicative', period=found_period)
    result.plot()
    plt.show()


def linear_trend(x, m, q):
    return np.array(m * x + q)


def RMSE(y_actual, y_predicted):
    return np.sqrt(np.mean((y_predicted - y_actual) ** 2))


def fit_trend_model(museum):
    x = np.linspace(0, 73, 74)
    initial_guess = [-1, 1]
    data = museum.Visitors.to_numpy()

    best_params, _ = curve_fit(linear_trend, x, data, p0=initial_guess)
    print("I migliori parametri per la retta di regressione sono: m={0}, q={1}".format(round(best_params[0], 2),
                                                                                       round(best_params[1], 2)))
    yfit = linear_trend(x, best_params[0], best_params[1])
    rmse = RMSE(data, yfit)
    print("La loss calcolata tramite RMSE è pari a {}".format(round(rmse, 3)))

    plt.plot(data)
    plt.plot(yfit, label="Trend lineare decrescente")
    plt.title("Individuazione del trend")
    plt.legend()
    plt.show()


def find_seasonality(museum):
    data = museum.Visitors.to_numpy()
    max_pearson = {"index": 0, "pearson": 0}

    # Il primo elemento si otterrebbe con pearsonr(data[24:], data[24:]), cioè confrontando tra loro
    # gli stessi identici dati.
    pearson_indexes = [1]

    for i in range(1, 24):
        tmp, _ = pearsonr(data[24:], data[24-i: -i])
        pearson_indexes.append(tmp)
        # Si considera un valore superiore a 0.7 come indice di forte correlazione
        if abs(tmp) > max_pearson["pearson"] and abs(tmp) > 0.7:
            max_pearson["index"] = i
            max_pearson["pearson"] = tmp

    plt.title("Individuazione della stagionalità tramite l'indice di Pearson")
    x = np.linspace(0, len(pearson_indexes), len(pearson_indexes))
    barlist = plt.bar(x, pearson_indexes)
    barlist[max_pearson["index"]].set_color("r")
    plt.text(max_pearson["index"], 0.95, "Valore massimo: {0}\nIndice: {1}".format(round(max_pearson["pearson"], 2), max_pearson["index"]),
             fontsize=8, color="red", ha="center")
    plt.show()

    return max_pearson


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

    # Ricerca della stagionalità tramite l'indice di Pearson
    seasonality = find_seasonality(avila_adobe_visitors)

    # Decomposizione dei dati: vengono mostrati il trend, la stagionalità e i residui
    plot_seasonal_decomposition(avila_adobe_visitors, seasonality["index"])

    # Individuazione del trend
    fit_trend_model(avila_adobe_visitors)


'''
PARTE 1
- Trovare funzione di trend e parameter fitting - DONE
- Eliminare trend - 
- Determinare se utilizzare un modello moltiplicativo o additivo -
- Individuare stagionalità - DONE
- Destagionalizzare - 
- Fare previsione con funzione di trend - 

PARTE 2
- Fare previsione con modelli predittivi statistici e neurali
- Analizzare e confrontare i risultati
'''
