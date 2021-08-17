# neural_network
This is a one-hidden-layer neural network for binary classification using batch gradient descent.

## Dataset
The real dataset used is a spam classification data set, where each instance is an email message represented by 57 features and is associated with a binary label indicating whether the email is spam (+1) or non-spam (−1). 

The data set is divided into training and test sets; there are 250 training examples and 4351 test examples. 

The goal is to learn from the training set a classifier that can classify new email messages in the test set.

## About
The algorithm is applied to both the synthetic dataset and the read spam dataset. 

For each dataset, train the model with both a logistic sigmoid activation function and a ReLU activation function.
