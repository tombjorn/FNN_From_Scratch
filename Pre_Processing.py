import pandas as pd
import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt 


class Process():
    def __init__(self, df):
        self.column_names = ['x_pos', 'y_pos', 'x_vel', 'y_vel']
        self.df = df
        self.df.columns = self.column_names

    def min_max_normalisation(self):
        self.min = self.df.min()
        self.max = self.df.max()

        self.df_min_max = pd.DataFrame()
        for col in self.column_names:
            self.df_min_max[col] = self.df[col].apply(lambda x: (x - self.min[col]) / (self.max[col] - self.min[col]))
        return self.df_min_max
    
    def z_score_normalisation(self):
        self.df_z_score = pd.DataFrame()
        for col in self.column_names:
            self.df_z_score[col] = self.df[col].apply(lambda x: (x - self.df.mean()[col] / self.df.std()[col]))
        print(f"Z_SCORE : {self.df_z_score.describe()}")
        return self.df_z_score
    
=======
import matplotlib.pyplot as plt

# ! remember to rearrange outputs


class Process():

    def sort_columns(self, df):
        column_names = ['x_pos', 'y_pos', 'y_vel', 'x_vel']
        df.columns = column_names
        df = df.reindex(columns=['x_pos', 'y_pos', 'x_vel', 'y_vel'])
        return df

    def min_max_normalisation(self, df):
        min = df.min()
        max = df.max()

        df_min_max = pd.DataFrame()
        for col in df.columns.tolist():
            df_min_max[col] = df[col].apply(lambda x: (
                x - min[col]) / (max[col] - min[col]))
        return df_min_max

    def z_score_normalisation(self):
        self.df_z_score = pd.DataFrame()
        for col in self.column_names:
            self.df_z_score[col] = self.df[col].apply(
                lambda x: (x - self.df.mean()[col] / self.df.std()[col]))
        print(f"Z_SCORE : {self.df_z_score.describe()}")
        return self.df_z_score

    def feature_label_split(self, df):
        features = df.loc[:, :'y_pos'].to_numpy()
        labels = df.loc[:, 'x_vel':].to_numpy()
        print(
            f'features shape : {features.shape}, labels shape : {labels.shape}')
        return features, labels

>>>>>>> 778605b (rewrote neural_net, adds update to pre_processing, grid search and testing script)
    def split_data(self, df):
        training = df.sample(frac=0.7, random_state=1)
        concat_1 = pd.concat([df, training])
        test_and_validation = concat_1.drop_duplicates(keep=False)
        testing = test_and_validation.sample(frac=0.5, random_state=1)
        concat_1 = pd.concat([test_and_validation, testing])
        validation = concat_1.drop_duplicates(keep=False)
        return training, validation, testing
<<<<<<< HEAD
    

def plot_RMSE(history, network_shape, LR, lamb):
    n_epochs = history['epochs']
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, history['epochs'], 1), history['training_rmse'],label='Training RMSE', color='blue')
    ax.plot(np.arange(0, history['epochs'], 1), history['validation_rmse'], label='Validation RMSE', color='red')
    ax.axvline(x = history['optimum_epoch'], color = 'orange', label = 'optimum')
    plt.title(f'Shape : 2-{network_shape[0]}-2, Epochs : {n_epochs} - Learning Rate : {LR}, Lambda : {lamb}')
    plt.legend()
    plt.show()
=======


def plot_predictions(predictions, y_test):
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(predictions), 1), predictions,
            label='Predictions', color='blue')
    ax.plot(np.arange(0, len(y_test), 1), y_test,
            label='Ground Truth', color='red')
    plt.legend()
    plt.show()


def plot_RMSE_2(training, validation, epochs, history, n_neurons, LR, alpha):
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, epochs, 1),
            training, label='Training RMSE', color='blue')
    ax.plot(np.arange(0, epochs, 1),
            validation, label='Validation RMSE', color='red')
    ax.axvline(x=history['optimum_epoch'], linestyle='dashed',
               color='black', label=f'optimum: {history["global_minimum"]:.4f}')
    plt.title(
        f'n_neurons : {n_neurons}, Epochs : {epochs} - Learning Rate : {LR}, alpha : {alpha}')
    plt.legend()
    plt.show()


def plot_RMSE(history, n_neurons, LR, alpha, show_optimum=False):
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, history['total_epochs'], 1),
            history['rmse_train'], label='Training RMSE', color='blue')
    ax.plot(np.arange(0, history['total_epochs'], 1),
            history['rmse_val'], label='Validation RMSE', color='red')
    if show_optimum:
        ax.axvline(x=history['optimum_epoch'], linestyle='dashed',
               color='black', label=f'optimum: {history["global_minimum"]:.4f}')
    plt.title(
        f'n_neurons : {n_neurons}, Epochs : {history['total_epochs']} - Learning Rate : {LR}, alpha : {alpha}')
    plt.legend()
    plt.show()
>>>>>>> 778605b (rewrote neural_net, adds update to pre_processing, grid search and testing script)
