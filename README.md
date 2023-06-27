# Handwritten Digit Recognition
This program uses a two-layer neural network (i.e, one hidden-layer) to perform the handwritten digit recognition task

# Neural Network Structure: 
The neural network will have 784 inputs, one hidden layer with n hidden units (where n is a parameter), and 10 output units. The hidden and output units use the sigmoid activation function. The network is fully connected —that is, every input unit connects to every hidden unit, and every hidden unit connects to every output unit. Every hidden and output unit also has a weighted connection from a bias unit, whose value is set to 1.

# Network Training:  
Uses back-propagation with stochastic gradient descent to train the network.

# Preprocessing: 
Data values are scaled to be between 0 and 1 by dividing by 255. 
Initial weights: The network starts off with small (−.05< w < .05) random positive and negative weights.
