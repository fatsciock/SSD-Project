import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.seasonal import seasonal_decompose
import os


def plot_data(data):
    visitors_before_covid = data.iloc[:74]
    visitors_during_covid = data.iloc[73:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Grafici dei dati con e senza le misurazioni nel periodo COVID-19")

    ax1.plot(visitors_before_covid.Visitors, label="Misurazioni in assenza del COVID-19")
    ax1.plot(visitors_during_covid.Visitors, 'r-.', label="Misurazioni durante il periodo COVID-19")
    ax1.legend()

    # Grafico dei valori dei visitatori al museo Avila Adobe che verranno utilizzati da qui in avanti
    ax2.plot(visitors_before_covid.Visitors, label="Visitatori del museo Avila Adobe")
    ax2.legend()

    plt.show()


def plot_seasonal_decomposition(data):
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    result = seasonal_decompose(data.Visitors, model='multiplicative', period=12)
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

    # Decomposizione dei dati: vengono mostrati il trend, la stagionalità e i residui
    plot_seasonal_decomposition(avila_adobe_visitors)

    # Individuazione del trend
    fit_trend_model(avila_adobe_visitors)



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
