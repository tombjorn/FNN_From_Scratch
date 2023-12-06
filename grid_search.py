from net import Network
from Pre_Processing import Process
import pandas as pd
import numpy as np

data = pd.read_csv('ce889_dataCollection.csv')
process = Process()

data = process.sort_columns(data)
data = process.min_max_normalisation(data)

training, validation, testing = process.split_data(data)

x_train, y_train = process.feature_label_split(training)
x_val, y_val = process.feature_label_split(validation)
x_test, y_test = process.feature_label_split(testing)


learning_rate = [0.001, 0.1, 0.3, 0.5, 0.7, 0.8]
alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
n_neurons = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]

epochs = 50
optimum_validation = 1.
best_LR = 0
best_N = 0
best_A = 0
best_history = dict()
for lr in learning_rate:
    for n in n_neurons:
        for a in alpha:
            print(lr, n, a)
            network = Network(h_neurons=n, alpha=a,
                              learning_rate=lr, patience=10)
            network.init_weights()
            history = network.fit(x_train, y_train, x_val,
                                  y_val, epochs=epochs, early_stopping=True)
            if 'global_minimum' not in history:
                validation = history['final_val']
            else:
                validation = history['global_minimum']

            if validation < optimum_validation:
                optimum_validation = validation
                best_LR = lr
                best_N = n
                best_A = a
                best_history = history

# Save optimum or final weights
print(history.keys())
# TODO Check early stopping is working as im worried now.
if 'optimum_weights' not in best_history:
    weights = best_history['final_weights']
    validation = best_history['final_val']
else:
    weights = best_history['optimum_weights']
    validation = best_history['global_minimum']

print(f'Learning Rate : {best_LR}, No. Neurons : {best_N}, Alpha : {best_A}')
print(f'Validation : {validation}')
np.savez(f'n_neurons-{best_N}, LR-{best_LR}, alpha-{best_A}-optimum_weights.npz',
         hidden=weights[0], output=weights[1])
