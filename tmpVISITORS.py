import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import datetime
import os


def difference(data, interval):
    return np.array([data[i] - data[i - interval] for i in range(interval, len(data))])


def invert_difference(original_data, diff_data, interval):
    return np.array(
        [diff_data[i - interval] + original_data[i - interval] for i in range(interval, len(original_data))])


os.chdir(os.path.dirname(os.path.abspath(__file__)))

total_visitors = pd.read_csv("Data/museum-visitors.csv", index_col=0)

if False:
    cont = 0
    for col in total_visitors:
        plt.plot(total_visitors.loc[:, col], label=col)
        cont += 1
        if cont == 3:
            plt.legend()
            plt.show()
            cont = 0

    plt.legend()
    plt.show()

visitors = total_visitors.iloc[:74, 1]
'''
visitors = visitors.loc[:, ["Date", "Total"]]
visitors = visitors.sort_values(by="Date")
visitors = visitors.groupby("Date")["Total"].sum().reset_index()
'''

plt.plot(visitors, label="original")
plt.legend()
plt.show()

diff = difference(visitors, 1)
plt.plot(diff, label="diff")
plt.legend()
plt.show()

log = np.log(visitors)
plt.plot(log, label="log")
plt.legend()
plt.show()

logdiff = difference(log, 1)
plt.plot(logdiff, label="logdiff")
plt.legend()
plt.show()

## PARTE 1 ##
# Trovare funzione di trend e parameter fitting
# Eliminare trend e determinare se utilizzare un modello moltiplicativo o additivo
# Individuare stagionalità e destagionalizzare
# Fare previsione con funzione di trend

## PARTE 2 ##
# Fare previsione con modelli predittivi statistici e neurali
# Analizzare e confrontare i risultati
