import os
import pandas as pd
from plot_functions import plot_data, plot_all_museums
from utility_functions import *
from trend_season_functions import run_TREND_SEASON
from SARIMA import run_SARIMA
from LSTM import run_LSTM
from MLP import run_MLP
from RANDOM_FOREST import run_RANDOM_FOREST

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
    periods_to_forecast = 24

    # Estensione delle date per includere i valori predetti
    extended_dates = extend_dates(museum_visitors, periods_to_forecast)

    # Creazione del modello tramite ricerca del trend, stagionalità e relativi coefficienti
    run_TREND_SEASON(museum_visitors, number_of_measurements, periods_to_forecast, extended_dates)

    # Algoritmo predittivo statistico
    # check_stationarity(museum_visitors)
    run_SARIMA(number_of_measurements, museum_visitors, extended_dates, periods_to_forecast)

    # Algoritmi predittivo neurali
    MLP_predictions = run_MLP(museum_visitors, extended_dates, periods_to_forecast)
    LSTM_predictions = run_LSTM(museum_visitors, extended_dates, periods_to_forecast)

    # Algoritmo Machine Learning
    RF_predictions = run_RANDOM_FOREST(museum_visitors, extended_dates, periods_to_forecast)

    print("-------Diebold-Mariano Test-------")

    # Confronto MLP e LSTM
    diebold_mariano(museum_visitors.Visitors[12:], MLP_predictions, LSTM_predictions, "MLP", "LSTM")

    # Confronto RF e MLP
    diebold_mariano(museum_visitors.Visitors[12:], RF_predictions, MLP_predictions, "RF", "MLP")

    # Confronto RF e LSTM
    diebold_mariano(museum_visitors.Visitors[12:], RF_predictions, LSTM_predictions, "RF", "LSTM")

'''
PARTE 1
- Trovare funzione di trend e parameter fitting - DONE
- Eliminare trend - DONE
- Determinare se utilizzare un modello moltiplicativo o additivo - DONE
- Individuare stagionalità - DONE
- Destagionalizzare - DONE
- Fare previsione con funzione di trend - DONE  

PARTE 2
- Fare previsione con modelli predittivi statistici e neurali - DONE
- Analizzare e confrontare i risultati - DONE
'''
