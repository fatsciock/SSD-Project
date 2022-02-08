import os
import pandas as pd
from plot_functions import *
from utility_functions import *
from trend_season_functions import *
from statiscal_algorithm import *
from neural_algorithms import *

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
    period_to_predict = 24
    len_to_predict = number_of_measurements + period_to_predict
    x_predict = np.linspace(0, len_to_predict - 1, len_to_predict)
    y_predict = linear_trend(x_predict, regression_params[0], regression_params[1])

    # Estensione delle date per includere i valori predetti
    extended_dates = extend_dates(museum_visitors, period_to_predict)

    predicted = []
    for i in range(len(x_predict)):
        predicted.append(y_predict[i] * season_coeff[i % seasonality])

    # Plot della previsione effettuata
    plot_prediction(museum_visitors, trend_season, predicted, y_predict, extended_dates)

    # Calcolo dell'errore commesso dal modello
    rmse = RMSE(museum_visitors.Visitors.to_numpy(), np.array(trend_season))
    print("La loss del modello calcolata tramite RMSE è pari a {}\n".format(round(rmse, 3)))

    # Algoritmo predittivo statistico

    # check_stationarity(museum_visitors)

    # run_statistical_algorithm(number_of_measurements, museum_visitors, extended_dates)

    # Algoritmo predittivo neurale
    run_neural_algorithms(museum_visitors, seasonality, extended_dates)

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
