# forecast monthly births with random forest
import numpy as np
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainY = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=700)
    model.fit(trainX, trainY)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = random_forest_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    err = mean_squared_error(test[:, -1], predictions)
    return error, err, test[:, -1], predictions


# load the dataset
all_visitors = read_csv("Data/museum-visitors.csv", parse_dates=["Month"], index_col=0)
values = all_visitors['Avila Adobe'].values[:74]
# transform the time series data into supervised learning
look_back = 12
data = series_to_supervised(values, n_in=look_back)

cutpoint = int(0.8 * len(values))

train, test = train_test_split(data, len(values) - cutpoint)

train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

model = RandomForestRegressor(n_estimators=500)
model.fit(train_x, train_y)

predictions_train = model.predict(train_x)
predictions_test = model.predict(test_x)


X = np.linspace(0, len(values) - 1, len(values))

# evaluate
# mae, mse, y, yhat = walk_forward_validation(data, len(values) - cutpoint)
# print('MAE: %.3f' % mae)
# print('RMSE: %.3f' % np.sqrt(mse))
# plot expected vs predicted
pyplot.plot(X, values, label='Expected')
pyplot.plot(X[:cutpoint-look_back], predictions_train, label='Predicted')
pyplot.plot(X[cutpoint-look_back:len(data)], predictions_test, label='Predicted')
pyplot.legend()
pyplot.show()
