# Brandon Gatewood
# CS 445 ML
# Programming 1

# This project implements a two-layer neural network to perform the handwritten
# digit recognition

import csv
import numpy as np
from scipy.special import expit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


class NeuralNetwork:
    bias = 1
    learning_rate = 0.1
    epochs = 50

    # Experiment 1
    # n = 20
    # n = 50
    n = 100

    # Experiment 2
    # momentum = 0.9
    # momentum = 0
    # momentum = 0.25
    momentum = 0.5

    # Initialize network weights to small random numbers between [-0.05, 0.05]
    # Weight from hidden layer to output layer
    wt_hl_to_ol = np.random.uniform(-0.05, 0.05, (n + 1, 10))
    # Weight from the input layer to hidden layer
    wt_il_to_hl = np.random.uniform(-0.05, 0.05, (785, n))

    # Track the previous delta weights to use during back propagation
    # Previous delta weight from hidden layer to output layer
    prv_wt_hl_to_ol = np.zeros((n + 1, 10))
    # Previous delta weight from input layer to hidden layer
    prv_wt_il_to_hl = np.zeros((785, n))

    # Track the activations
    input_hl = np.zeros((1, n + 1))
    input_hl[0, 0] = 1

    def __init__(self, test_dataset, train_dataset):
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset

    def run(self):
        train_accuracy = np.zeros(self.epochs)
        test_accuracy = np.zeros(self.epochs)

        for epoch in tqdm(range(self.epochs)):
            train_accuracy[epoch] = self.two_layered_perceptron(epoch, self.train_dataset, 1)
            test_accuracy[epoch] = self.two_layered_perceptron(epoch, self.test_dataset, 0)

        # Plot training accuracy and testing accuracy
        plt.plot(test_accuracy, label='Testing Set')
        plt.plot(train_accuracy, label='Training Set')
        plt.ylabel("Accuracy in %")
        plt.ylim(0, 100)
        plt.xlabel("Epoch")
        img = "50"
        # plt.title("Experiment 1: hidden units = " + str(self.n))
        plt.title("Experiment 2: momentum = " + str(self.momentum))
        # plt.title("Experiment 3: 1/4 of training")
        # plt.title("Experiment 3: 1/2 of training")
        plt.savefig(img)
        plt.legend()
        plt.show()

    def two_layered_perceptron(self, epoch, dataset, train_flag):
        # def two_layered_perceptron(self, epoch, dataset, train_flag):
        # Track the predicted and actual target output
        predicted_list = []
        target_list = []

        # Present input to the input layer
        for i in range(dataset.shape[0]):
            # Track each target output in the data set
            target = dataset[i][0].astype('int')
            target_list.append(target)

            # Scale weights and set bias
            xi = dataset[i].astype('float16') / 255
            xi[0] = self.bias
            xi = xi.reshape(1, 785)

            # Forward propagate the activations times the weights to each node in
            # the hidden layer. Then Apply sigmoid function to sum of weights times
            # inputs to each hidden unit
            dp_hl = np.dot(xi, self.wt_il_to_hl)
            sig_hl = expit(dp_hl)
            self.input_hl[0, 1:] = sig_hl

            # Forward propagate the activations time the weights from the hidden layer
            # to the output layer. Then apply sigmoid function to sum of weights times
            # inputs to each output unit.
            dp_ol = np.dot(self.input_hl, self.wt_hl_to_ol)
            sig_ol = expit(dp_ol)

            # Interpret the output layer as a classification and track it
            predicted = np.argmax(sig_ol)
            predicted_list.append(predicted)

            # Back propagation algorithm one layer at a time to update all weights
            # in the network. Train flag to determine if dataset is being trained
            # or tested
            if train_flag == 1 and epoch > 0:
                # Determine the error term for each output unit
                # Ground truth value for the kth value of the label
                t_k = np.zeros((1, 10)) + 0.1
                t_k[0, target] = 0.9

                # Compute the error term for the output unit
                err_term_ol = sig_ol * (1 - sig_ol) * (t_k - sig_ol)

                # Compute the error term for the hidden unit
                err_term_hl = sig_hl * (1 - sig_hl) * np.dot(err_term_ol, self.wt_hl_to_ol[1:, :].T)

                # Update output layer and previous delta weight
                delta_wt_ol = (self.learning_rate * err_term_ol * self.input_hl.T) + (self.momentum * self.prv_wt_hl_to_ol)
                self.prv_wt_hl_to_ol = delta_wt_ol
                self.wt_hl_to_ol = self.wt_hl_to_ol + delta_wt_ol

                # Update hidden layer and previous delta weight
                delta_wt_hl = (self.learning_rate * err_term_hl * xi.T) + (self.momentum * self.prv_wt_il_to_hl)
                self.prv_wt_il_to_hl = delta_wt_hl
                self.wt_il_to_hl = self.wt_il_to_hl + delta_wt_hl

        # Compute accuracy of the model
        accuracy = (np.array(predicted_list) == np.array(target_list)).sum() / float(len(target_list)) * 100

        # Compute confusion matrix for testing dataset
        # if train_flag == 0:
        if epoch == self.epochs - 1 and train_flag == 0:
            # print("\nConfusion Matrix of testing set for hidden units = " + str(self.n))
            print("\nConfusion Matrix of testing set for momentum =  " + str(self.momentum))
            # print("\nConfusion Matrix of testing set for quarter training")
            # print("\nConfusion Matrix of testing set for half training")
            cfm = confusion_matrix(target_list, predicted_list)
            print(cfm)
            # diagonal_sum = sum(np.diag(cfm))

        return accuracy


# load training and testing dataset into arrays
f = open("mnist_test.csv", 'r')
test_data = csv.reader(f)
test = np.array(list(test_data))

f = open("mnist_train.csv", 'r')
train_data = csv.reader(f)
train = np.array(list(train_data))

# Experiment 3
# quarter = 15000
# half = 30000
# np.random.shuffle(train)
# train = train[0:quarter]
# train = train[0:half]

neural_network = NeuralNetwork(test, train)
neural_network.run()
