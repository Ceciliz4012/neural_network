# neural_network
This is a one-hidden-layer neural network for binary classification using batch gradient descent.

## Dataset
The real dataset used is a spam classification data set, where each instance is an email message represented by 57 features and is associated with a binary label indicating whether the email is spam (+1) or non-spam (−1). 

The data set is divided into training and test sets; there are 250 training examples and 4351 test examples. 

The goal is to learn from the training set a classifier that can classify new email messages in the test set.

## About
The algorithm is applied to both the synthetic dataset and the read spam dataset. 

For each dataset, trained the model with both a logistic sigmoid activation function and a ReLU activation function.

In each case, considered d1 in the range {1, 5, 10, 15, 25, 50}, and performed 5-fold cross validation to select d1 from this range. Then, created a plot of the training, test, and cross-validation errors as a function of the number of hidden units d1.

Data analysis and visualization are in the analysis.pdf file.
