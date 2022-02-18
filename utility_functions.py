import numpy as np
import pandas as pd
from diebold_mariano_test import dm_test
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from plot_functions import plot_autocorrelation


def get_visitors_of(museum_name, all_visitors):
    result = all_visitors.loc[:, [museum_name]]

    # Si rinomina la colonna relativa ai visitatori del museo e si imposta quella delle date come indice
    result.rename(columns={museum_name: 'Visitors'}, inplace=True)

    return result


def RMSE(y_actual, y_predicted):
    return np.sqrt(np.mean((y_predicted - y_actual) ** 2))


def check_mean(t1, t2):
    print("\nMedia prima metà dei dati: {0}\n"
          "Media seconda metà dei dati: {1}".format(round(t1.mean()), round(t2.mean())))


def check_variance(t1, t2):
    print("\nVarianza prima metà dei dati: {0}\n"
          "Varianza seconda metà dei dati: {1}".format(round(t1.var()), round(t2.var())))


def adf_test(data):
    adf_value, pvalue, _, _, critical_values, _ = adfuller(data)
    print("\nVerifica stazionarietà con ADF:\n"
          "ADF: {0}\n"
          "p-value: {1}\nCritical values:".format(round(adf_value, 3), round(pvalue, 3)))

    for k, v in critical_values.items():
        print("\t{0}: {1}".format(k, round(v, 3)))

    print("Serie non stazionaria" if pvalue > 0.05 else "Serie stazionaria")


def kpss_test(data):
    kpss_value, p_value, _, critical_values = kpss(data, nlags="auto")
    print("\nVerifica stazionarietà con KPSS:\n"
          "KPSS: {0}\n"
          "p-value: {1}\nCritical values:".format(round(kpss_value, 3), round(p_value, 3)))

    for k, v in critical_values.items():
        print("\t{0}: {1}".format(k, round(v, 3)))

    print("Serie stazionaria" if p_value > 0.05 else "Serie non stazionaria")


def check_stationarity(museum):
    data = museum.Visitors.to_numpy()
    plot_autocorrelation(museum)

    cutpoint = int(len(data) / 2)
    t1 = data[:cutpoint]
    t2 = data[cutpoint:]
    check_mean(t1, t2)
    check_variance(t1, t2)

    adf_test(data)
    kpss_test(data)


def print_loss(yfore, ypred, data, cut):
    rmse_pred = RMSE(data[:cut], np.array(ypred))
    rmse_fore = RMSE(data[cut:], np.array(yfore))
    print("La loss del modello SARIMA calcolata tramite RMSE è:\n"
          "train: {0}\ntest: {1}".format(round(rmse_pred, 3), round(rmse_fore, 3)))


def extend_dates(museum_visitors, periods_to_extend):
    all_dates = museum_visitors.index
    start_date = all_dates[0]
    periods = periods_to_extend + len(all_dates)
    extended_dates = pd.date_range(start=start_date, periods=periods, freq='MS').tolist()
    return extended_dates


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def diebold_mariano(data, pred1, pred2, name1, name2):
    dm_result = dm_test(data, pred1, pred2)
    print("\nRisultato del test Diebold-Mariano confrontando le previsioni di {} e di {}: \nDM={} \np_value={}"
          .format(name1, name2, round(dm_result.DM, 4), round(dm_result.p_value, 4)))
    print("a = 0.05, z-score(a/2) = 1.96")
    print("HO rifiutata: le 2 previsioni non hanno la stessa accuratezza\n" if np.abs(dm_result.DM) > 1.96
          else "HO non rifiutata: le 2 previsioni hanno la stessa accuratezza")
