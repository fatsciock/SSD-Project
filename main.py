import os
import pandas as pd
from plot_functions import plot_data, plot_all_museums
from utility_functions import *
from trend_season_functions import run_TREND_SEASON
from SARIMA import run_SARIMA
from LSTM import run_LSTM
from MLP import run_MLP

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    all_visitors = pd.read_csv("Data/museum-visitors.csv", parse_dates=["Month"], index_col=0)

    # plot_all_museums(museum_visitors)

    # Si è scelto di lavorare sui visitatori del museo 'Avila Adobe'
    museum_visitors = get_visitors_of("Avila Adobe", all_visitors)

    plot_data(museum_visitors)

    # Esclusione dati del periodo COVID
    museum_visitors = museum_visitors.iloc[:74]
    number_of_measurements = len(museum_visitors.Visitors)
    period_to_predict = 24

    # Estensione delle date per includere i valori predetti
    extended_dates = extend_dates(museum_visitors, period_to_predict)

    # Creazione del modello tramite ricerca del trend, stagionalità e relativi coefficienti
    run_TREND_SEASON(museum_visitors, number_of_measurements, period_to_predict, extended_dates)

    # Algoritmo predittivo statistico
    # check_stationarity(museum_visitors)
    run_SARIMA(number_of_measurements, museum_visitors, extended_dates)

    # Algoritmo predittivo neurale
    run_MLP(museum_visitors, extended_dates)
    run_LSTM(museum_visitors, extended_dates)

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
