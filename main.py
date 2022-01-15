import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy.optimize import curve_fit
import os


def plot_data(data):
    visitors_before_covid = data.iloc[:74]
    visitors_during_covid = data.iloc[73:]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(15)
    fig.set_figheight(6)
    fig.suptitle("Grafici dei dati con e senza le misurazioni nle periodo COVID-19")

    ax1.plot(visitors_before_covid, label="Misurazioni in assenza del COVID-19")
    ax1.plot(visitors_during_covid, 'r-.', label="Misurazioni durante il periodo COVID-19")
    ax1.legend()

    # Grafico dei valori dei visitatori al museo Avila Adobe che verranno utilizzati da qui in avanti
    ax2.plot(avila_adobe_visitors.iloc[:74], label="Visitatori del museo Avila Adobe")
    ax2.legend()
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    museum_visitors = pd.read_csv("Data/museum-visitors.csv")

    # Si è scelto di lavorare sui visitatori del museo 'Avila Adobe'
    museum_name = "Avila Adobe"
    avila_adobe_visitors = museum_visitors.loc[:, ["Month", museum_name]]

    # Qui si è modificato il formato della data, in modo da renderlo meglio leggibile
    avila_adobe_visitors['Month'] = pd.to_datetime(avila_adobe_visitors["Month"],
                                                   format="%Y-%d-%mT%H:%M:%S.%f").dt.strftime('%d/%m/%Y')

    # Infine si è impostata la colonna contenente le date come indice del DataFrame
    avila_adobe_visitors.set_index('Month', inplace=True)

    # Ora vengono mostrati i dati in un grafico che evidenzia i valori misurati durante la pandemia
    # del COVID-19. Tali dati non verranno usati per tutte le elaborazioni successive.
    # Infatti viene anche mostrato un grafico dei dati esenti dagli effetti della pandemia
    plot_data(avila_adobe_visitors)

    # Esclusione dati del periodo COVID
    avila_adobe_visitors = avila_adobe_visitors.iloc[:74]


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
