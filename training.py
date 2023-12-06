from net import Network
from Pre_Processing import Process, plot_RMSE, plot_predictions
import pandas as pd
import numpy as np
import time

data = pd.read_csv('ce889_dataCollection.csv')
process = Process()

data = process.sort_columns(data)
data = process.min_max_normalisation(data)

training, validation, testing = process.split_data(data)

x_train, y_train = process.feature_label_split(training)
x_val, y_val = process.feature_label_split(validation)
x_test, y_test = process.feature_label_split(testing)


learning_rate = 0.8
n_neurons = 18
alpha = 0.8
epochs = 100

network = Network(h_neurons=n_neurons, alpha=0.8,
                  learning_rate=0.8, patience=10)
network.init_weights()

history = network.fit(x_train, y_train, x_val, y_val,
                      epochs=epochs, early_stopping=True)

plot_RMSE(history, n_neurons, learning_rate, alpha, show_optimum=False)
