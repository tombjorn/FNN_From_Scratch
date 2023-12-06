import numpy as np
from net import Network
from Pre_Processing import Process, plot_RMSE, plot_predictions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def R2_score(truth, prediction):
    mean_truth = sum(truth) / len(truth)
    ss_total = sum(((y - mean_truth) ** 2) for y in truth)

    ss_res = sum(((truth[i] - prediction[i]) ** 2) for i in range(len(truth)))
    R2 = 1 - (ss_res / ss_total)
    return R2


loaded_arrays = np.load('n_neurons-20, LR-0.8, alpha-0.6-optimum_weights.npz')
hidden = loaded_arrays['hidden']
output = loaded_arrays['output']


data = pd.read_csv('ce889_dataCollection.csv')
process = Process()

data = process.sort_columns(data)
data = process.min_max_normalisation(data)

training, validation, testing = process.split_data(data)

x_train, y_train = process.feature_label_split(training)
x_val, y_val = process.feature_label_split(validation)
x_test, y_test = process.feature_label_split(testing)


network = Network(h_neurons=20, alpha=0.6, learning_rate=0.8, patience=15)
network.load_weights([hidden, output])

predictions = [network.calculate_output(x) for x in x_test]
y_A1 = [x[0] for x in x_test]
y_A2 = [x[1] for x in x_test]
p_A1 = [x[0] for x in predictions]
p_A2 = [x[1] for x in predictions]

network = Network(h_neurons=16, alpha=0.6, learning_rate=0.01, patience=15)
network.init_weights()
network.fit(x_train, y_train, x_val, y_val, epochs=100, early_stopping=True)
network.save_weights()
predictions = [network.calculate_output(x) for x in x_test]
y_B1 = [x[0] for x in x_test]
y_B2 = [x[1] for x in x_test]
p_B1 = [x[0] for x in predictions]
p_B2 = [x[1] for x in predictions]

print(f'optimum R2 = {R2_score(y_A1, p_A1)}')
print(f'trained R2 = {R2_score(y_B1, p_B1)}')
plt.scatter(range(len(p_A1)), p_A1, color='red', label='optimum prediction')
plt.scatter(range(len(p_B1)), p_B1, color='green', label='fitted prediction')
plt.scatter(range(len(p_A1)), y_A1, color='blue', label='ground truth')

plt.title('Predicted vs. True Values')
plt.legend()
plt.grid(True)
plt.show()
