# Assumes only 3 layers 2 in and 2 out

# TODO implement history object to be able to save weights
# TODO implement early stopping

import numpy as np


class Network():
    def __init__(self, h_neurons=4, alpha=0.6, learning_rate=0.001, patience=3):
        self.h_neurons = h_neurons
        self.weights = None
        self.h_output = None
        self.y_output = None

        self.lmbd = 0.8
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.prev_delta = []

        self.early_stop_count = 0
        self.best_validation = 1
        self.rolling_mean = None
        self.window_size = 5
        self.patience = 3

        self.history = {
            'weights': [],
            'validation': [],
            'epoch': []
        }

    def init_weights(self):
        """
        Initialises the weights of the network. Adds an extra row of weights 
        to simulate the bias for when the network back propagates.

        Returns:
            list: list containing the two weights matrices.
        """
        w_1 = np.random.rand(2 + 1, self.h_neurons)
        w_2 = np.random.rand(self.h_neurons + 1, 2)
        self.weights = [w_1, w_2]
        return self.weights

    def get_weights(self):
        return self.weights

    def load_weights(self, weights):
        """
        Load weights from a previously trained instance of the network.

        Args:
            weights (list): list containing two weights matrices
        """
        self.weights = weights

    def apply_activation(self, v):
        """
        Applys a list mapping using the sigmoid activation function

        Args:
            v (list): output of the dot multiplication of weights and inputs

        Returns:
            list: activation function output
        """
        output = list(map(lambda x: (1/(1 + np.exp(-self.lmbd * x))), v))
        return output

    def calculate_output(self, input_row):
        """
        Feeds forward through the network's layers.
        calculates the activation function outputs at each layer, and stores them
        as the instances attributes.

        Note - Appending 1 to the inputs of the layers in order to implement biases, which
        in conjunction with the 'extended' weights, allows for the biases to be updated during 
        back propagation.

        Args:
            input_row (list): list containing the rockets 2 features.

        Returns:
            list: list containing the two output's predictions
        """
        # input = [x1, x2]
        self.input_row = input_row
        x_1 = np.append(input_row, 1)
        # input_row = [x1, x2, 1]
        v_1 = np.dot(x_1, self.weights[0])
        self.h_output = self.apply_activation(v_1)
        # ? need to add an 'input row' of 1 to the next set of inputs, so we can use bias
        h_1 = np.append(self.h_output, [1])
        v_2 = np.dot(h_1, self.weights[1])
        self.y_output = self.apply_activation(v_2)
        return self.y_output

    def calculate_error(self, desired_row, predictions):
        """
        Given a ground truth and predicted value, calculates the error.

        Args:
            desired_row (list): one dimensional list containing ground truth output values
            predictions (list): one dimensional list containing predicted output values

        Returns:
            list: one dimensional list of errors
        """
        return desired_row - predictions

    def calculate_gradient(self, e):
        """
        Calculates the different gradients at different layers in the network.
        For the hidden layer's gradient, we need to multiply the normal equation by 
        the dot product of the node's weights and previous layer's gradients. 

        Args:
            e (list): one dimensional list of errors

        Returns:
            tuple: (hidden_gradient, local_gradient)
        """
        # e = [e1, e2]
        grad_loc = [self.lmbd * y * (1-y) * e[idx]
                    for idx, y in enumerate(self.y_output)]

        # ? including the bias weights is pointless as always going to be zero?
        grad_hid = [self.lmbd * h * (1-h) * np.dot(self.weights[1][idx], grad_loc)
                    for idx, h in enumerate(self.h_output)]

        return grad_hid, grad_loc

    def delta_weights(self, grad_hidden, grad_local):
        """
        Calculates the delta weights of the network. 

        Note - some confusion as to whether I keep the 1 value for the 'bias node' in 
        my 'self.*_output' variables (in calculate_output()) or just append it to 
        local variable when I need it. 

        Think if i keep it in, it will mess with my weight dot multiplication as I dont 
        want any weights going 'into' my bias nodes.

        Args:
            grad_hidden (list): list of hidden gradient values
            grad_local (list): list of local gradient values

        Returns:
            tuple: (delta_1, delta_2)
        """

        delta_1 = []
        delta_2 = []

        input_row = np.append(self.input_row, [1])

        for idx, node_weights in enumerate(self.weights[0]):
            node_deltas = np.array(grad_hidden) * \
                input_row[idx] * self.learning_rate
            delta_1.append(node_deltas)

        h_outputs = np.append(self.h_output, [1])
        for idx, node_weights in enumerate(self.weights[1]):
            node_deltas = np.array(grad_local) * \
                h_outputs[idx] * self.learning_rate
            delta_2.append(node_deltas)

        return delta_1, delta_2

    def update_weights(self, dw_1, dw_2):
        """
        Updates the weights by summing with the delta weights

        Args:
            dw_1 (list): delta weight matrix
            dw_2 (list): delta weight matrix

        Returns:
            list: list containing two weights matrices
        """

        w_1 = self.weights[0] + dw_1
        w_2 = self.weights[1] + dw_2
        self.weights = [w_1, w_2]
        return self.weights

    def predict(self, input_row):
        """
        Given an input, predicts the output.

        Args:
            input_row (list): input
        Note - Should only be used for the game, as it flips the outputs 
        back to the game's format (y, x).

        Returns:
            list: predicted output
        """
        prediction = self.calculate_output(input_row)
        return prediction[::-1]

    def forward_pass(self, input_row, desired_row):
        """
        Feed forard through the network given an initial input row.

        Args:
            input_row (list): [x1, x2]
            desired_row (list): [d1, d2]

        Returns:
            tuple: (prediction, error)
        """
        prediction = self.calculate_output(input_row)
        error = self.calculate_error(desired_row, prediction)
        return prediction, error

    def backwards_pass(self, error):
        """
        Back propagates through the network, given an output error.

        Args:
            error (list): [e1, e2]

        Returns:
            list: list containing two updated weights matrices
        """
        grad_hid, grad_loc = self.calculate_gradient(error)
        delta_1, delta_2 = self.delta_weights(grad_hid, grad_loc)

        if self.prev_delta != []:
            delta_1 = (self.alpha * self.prev_delta[0]) + delta_1
            delta_2 = (self.alpha * self.prev_delta[1]) + delta_2
        self.prev_data = [delta_1, delta_2]

        self.update_weights(delta_1, delta_2)
        return self.weights

    def rmse(self, error_list):
        """
        Calculates the rmse given a list of errors.

        Args:
            error_list (list): [[eX1, eY1], [eX2, eY2] .... [eXn, eYn]]

        Returns:
            float: RMSE value
        """
        # error_list[0] = [eX, eY]
        x_err = [error[0] for error in error_list]
        y_err = [error[1] for error in error_list]
        x_sqr = np.square(x_err)
        y_sqr = np.square(y_err)

        x_mse = np.mean(x_sqr)
        y_mse = np.mean(y_sqr)

        rmse_x = np.sqrt(x_mse)
        rmse_y = np.sqrt(y_mse)

        rmse = (rmse_x + rmse_y) / 2

        return rmse

    def shuffle(self, features, labels):
        """
        Shuffles the feature and label matrices while keeping their ordering coherant.
        Allowing the randomizing of inputs between epochs without disconnecting the features
        and labels

        Args:
            features (lsit): [x1, y1]
            labels (lsit): [x1, y1]

        Returns:
            tuple: (shuffled_features, shuffled_labels)
        """
        num = len(features)
        shuffle_idx = np.random.permutation(num)
        features = features[shuffle_idx]
        labels = labels[shuffle_idx]

        return features, labels

    def calculate_rolling_mean(self, validation_rmse):
        # Calculate rolling mean for validation RMSE
        if self.rolling_mean is None:
            self.rolling_mean = validation_rmse
        else:
            self.rolling_mean = (
                self.rolling_mean * (self.window_size - 1) + validation_rmse) / self.window_size

    def early_stop(self):
        """
        Updates the history dictionary every iteration and adds a count if 
        validation loss is less than the previous. Returns True if count is same
        value of self.patience, False otherwise.


        Args:
            current_epoch (int): current epoch number
            validation_rmse (float): validation rmse of latest epoch

        Returns:
            bool: True if early stop criteria met, False otherwise.
        """
        if self.rolling_mean is None:
            return False

        if self.rolling_mean > self.best_validation:
            self.early_stop_count += 1
            if self.early_stop_count >= self.patience:
                return True
        else:
            self.early_stop_count = 0
            self.best_validation = self.rolling_mean
            return False
        return False

    def fit(self, x_train, y_train, x_val, y_val, epochs=10, early_stopping=True):
        """
        Undergoes training the model with validation checks every epoch.

        Args:
            x_train (list): [x1, y1]
            y_train (list): [x1, y1]
            x_val (list): [x1, y1]
            y_val (list): [x1, y1]
            epochs (int, optional): total number of epochs to train for. Defaults to 10.
            early_stopping (bool, optional): bool to implement early stopping. Defaults to True.

        Returns:
            dict: history object with key/value pairs ->
                'weights': list, 
                'validation': float,
                'epoch': int,
                'rmse_train' : list,
                'rmse_val' : list

        """
        training_rmse_list = []
        validation_rmse_list = []
        for n in range(epochs):
            x_train, y_train = self.shuffle(x_train, y_train)
            x_val, y_val = self.shuffle(x_val, y_val)
            training_error = []
            validation_error = []

            # Training
            for idx, row in enumerate(x_train):
                prediction, error = self.forward_pass(row, y_train[idx])
                training_error.append(error)
                self.backwards_pass(error)

            # Validation
            for idx, row in enumerate(x_val):
                prediction, error = self.forward_pass(row, y_val[idx])
                validation_error.append(error)

            rmse_train = self.rmse(training_error)
            rmse_val = self.rmse(validation_error)
            # print(
            # f'Training RMSE : {rmse_train}, Validation RMSE : {rmse_val}')

            training_rmse_list.append(rmse_train)
            validation_rmse_list.append(rmse_val)

            self.calculate_rolling_mean(rmse_val)

            # TODO not sure whether to continue plotting to show the entire trainig, marking
            # TODO and storing the optimum, or literally stop and blue balls
            if self.early_stop():
                self.history['optimum_epoch'] = n + 1
                self.history['global_minimum'] = rmse_val
                self.history['optimum_weights'] = self.weights
                if early_stopping:
                    break
        self.history['rmse_train'] = training_rmse_list
        self.history['rmse_val'] = validation_rmse_list
        self.history['final_weights'] = self.weights
        self.history['final_val'] = rmse_val
        self.history['total_epochs'] = n+1
        return self.history

    def testing(self, x_test, y_test):
        """
        given some training features and labels, predict and calculate the testing RMSE.

        Args:
            x_test (list): [[x1, y1], [x2, y2]... [xn, yn]]
            y_test (list): [[x1, y1], [x2, y2]... [xn, yn]]

        Returns:
            tuple: (rmse, predictions)
        """
        predictions = []
        errors = []
        for idx, row in enumerate(x_test):
            predicted, error = self.forward_pass(row, y_test)
            predictions.append(predicted)
            errors.append(error)

        return self.rmse(errors), predictions

    def save_weights(self):
        np.savez(f'n_neurons-{self.h_neurons}, LR-{self.learning_rate}, alpha-{self.alpha}-optimum_weights.npz',
                 hidden=self.weights[0], output=self.weights[1])
