import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


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


def plot_all_museums(museum_visitors):
    for museum in museum_visitors:
        plt.plot(museum_visitors[museum], label=museum)

    plt.legend()
    plt.show()
