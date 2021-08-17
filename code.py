import numpy as np 
import matplotlib.pyplot as plt
import utils
import time
from neural_network import NeuralNetworkClassification


# Load synthetic data
d1_list = [1,5,10,15,25,50]
X_syn_train, y_syn_train, X_syn_test, y_syn_test = utils.load_all_train_test_data("/Users/ceciliz/Desktop/ps2_kit/P3/Synthetic-Dataset")
synthetic_folds = utils.load_all_cross_validation_data("/Users/ceciliz/Desktop/ps2_kit/P3/Synthetic-Dataset/CrossValidation")

# Synthetic sigmoid

sig_train_error = []
sig_test_error = []
cv_avg_error = []
training_time = []

for n in d1_list:
    print("running iteration for d1={}".format(n))

    initWeights = utils.load_initial_weights("P3/Synthetic-Dataset/InitParams/sigmoid/{}".format(n))

    model = NeuralNetworkClassification(X_syn_train.shape[1], num_hidden = n, activation = "sigmoid", W1=initWeights["W1"], b1=initWeights["b1"], W2=initWeights["W2"], b2=initWeights["b2"])

    s = time.perf_counter()
    model.fit(X_syn_train, y_syn_train, num_iters=8000, step_size=0.1)
    e = time.perf_counter()

    train_pred = model.predict(X_syn_train)
    test_pred = model.predict(X_syn_test)
    sig_train_error.append(utils.classification_error(train_pred, y_syn_train))
    sig_test_error.append(utils.classification_error(test_pred, y_syn_test))

    training_time.append(e - s)

    sum_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(synthetic_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr, num_iters=8000, step_size=0.1)
        pred = model.predict(X_lo)
        sum_error += utils.classification_error(pred, y_lo)

    cv_avg_error.append(sum_error / 5)


min_index = cv_avg_error.index(min(cv_avg_error))
print("Chosen d1: ")
print(d1_list[min_index])
print("Corresponding Training error: ")
print(sig_train_error[min_index])
print("Corresponding Test error: ")
print(sig_test_error[min_index])
print("Corresponding Training time: ")
print(training_time[min_index])

plt.plot(d1_list, sig_train_error, label = "training_error", color = 'forestgreen')
plt.plot(d1_list, sig_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(d1_list, cv_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.title("Sigmoid NN: Error vs. d1 for data 1")
plt.legend(loc = "upper right")
plt.show()


relu_train_error = []
relu_test_error = []
cv_avg_error = []
training_time = []

for n in d1_list:
    print("running iteration for d1={}".format(n))
    initWeights = utils.load_initial_weights("P3/Synthetic-Dataset/InitParams/relu/{}".format(n))

    model = NeuralNetworkClassification(X_syn_train.shape[1], num_hidden = n, activation = "relu", W1=initWeights["W1"], b1=initWeights["b1"], W2=initWeights["W2"], b2=initWeights["b2"])

    s = time.perf_counter()
    model.fit(X_syn_train, y_syn_train, num_iters=8000, step_size=0.01)
    e = time.perf_counter()

    train_pred = model.predict(X_syn_train)
    test_pred = model.predict(X_syn_test)
    relu_train_error.append(utils.classification_error(train_pred, y_syn_train))
    relu_test_error.append(utils.classification_error(test_pred, y_syn_test))
    training_time.append(e-s)

    sum_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(synthetic_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr, num_iters=8000, step_size=0.01)
        pred = model.predict(X_lo)
        sum_error += utils.classification_error(pred, y_lo)

    cv_avg_error.append(sum_error / 5)




min_index = cv_avg_error.index(min(cv_avg_error))
print("Chosen d1: ")
print(d1_list[min_index])
print("Corresponding Training error: ")
print(relu_train_error[min_index])
print("Corresponding Test error: ")
print(relu_test_error[min_index])
print("Corresponding Training time: ")
print(training_time[min_index])

plt.plot(d1_list, relu_train_error, label = "training_error", color = 'forestgreen')
plt.plot(d1_list, relu_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(d1_list, cv_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.title("Relu NN: Errors vs. d1 for data 1")
plt.legend(loc = "upper right")
plt.show()



#part b

X_train, y_train, X_test, y_test = utils.load_all_train_test_data("/Users/ceciliz/Desktop/ps2_kit/P3/Spam-Dataset")
real_folds = utils.load_all_cross_validation_data("/Users/ceciliz/Desktop/ps2_kit/P3/Spam-Dataset/CrossValidation")

d1_list = [1,5,10,15,25,50]

sig_train_error = []
sig_test_error = []
cv_avg_error = []
training_time = []

for n in d1_list:
    print("running iteration for d1={}".format(n))
    initWeights = utils.load_initial_weights("P3/Spam-Dataset/InitParams/sigmoid/{}".format(n))

    model = NeuralNetworkClassification(X_train.shape[1], num_hidden = n, activation = "sigmoid", W1=initWeights["W1"], b1=initWeights["b1"], W2=initWeights["W2"], b2=initWeights["b2"])

    s = time.perf_counter()
    model.fit(X_train, y_train, num_iters=8000, step_size=0.12)
    e = time.perf_counter()
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    sig_train_error.append(utils.classification_error(train_pred, y_train))
    sig_test_error.append(utils.classification_error(test_pred, y_test))
    training_time.append(e - s)

    sum_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(real_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr, num_iters=8000, step_size=0.12)
        pred = model.predict(X_lo)
        sum_error += utils.classification_error(pred, y_lo)

    cv_avg_error.append(sum_error / 5)


min_index = cv_avg_error.index(min(cv_avg_error))
print("Chosen d1: ")
print(d1_list[min_index])
print("Corresponding Training error: ")
print(sig_train_error[min_index])
print("Corresponding Test error: ")
print(sig_test_error[min_index])
print("Corresponding Training time: ")
print(training_time[min_index])

plt.plot(d1_list, sig_train_error, label = "training_error", color = 'forestgreen')
plt.plot(d1_list, sig_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(d1_list, cv_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.title("Sgimoid NN: Errors vs. d1 for real data")
plt.legend(loc = "upper right")
plt.show()


relu_train_error = []
relu_test_error = []
cv_avg_error = []
training_time = []

for n in d1_list:
    print("running iteration for d1={}".format(n))
    initWeights = utils.load_initial_weights("P3/Spam-Dataset/InitParams/relu/{}".format(n))
    model = NeuralNetworkClassification(X_train.shape[1], num_hidden = n, activation = "relu", W1=initWeights["W1"], b1=initWeights["b1"], W2=initWeights["W2"], b2=initWeights["b2"])

    s = time.perf_counter()
    model.fit(X_train, y_train, num_iters=8000, step_size=0.1)
    e = time.perf_counter()
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    relu_train_error.append(utils.classification_error(train_pred, y_train))
    relu_test_error.append(utils.classification_error(test_pred, y_test))
    training_time.append(e - s)

    sum_error = 0
    for i in range(5):
        lo_data, tr_data = utils.partition_cross_validation_fold(real_folds, i)
        (X_lo, y_lo) = lo_data
        (X_tr, y_tr) = tr_data
        model.fit(X_tr, y_tr, num_iters=8000, step_size=0.1)
        pred = model.predict(X_lo)
        sum_error += utils.classification_error(pred, y_lo)
    
    cv_avg_error.append(sum_error / 5)




min_index = cv_avg_error.index(min(cv_avg_error))
print("Chosen d1: ")
print(d1_list[min_index])
print("Corresponding Training error: ")
print(cv_train_error[min_index])
print("Corresponding Test error: ")
print(cv_test_error[min_index])
print("Corresponding Training time: ")
print(training_time[min_index])

plt.plot(d1_list, relu_train_error, label = "training_error", color = 'forestgreen')
plt.plot(d1_list, relu_test_error, label = "test_error", color = 'mediumvioletred')
plt.plot(d1_list, cv_avg_error, label = "avg_cv_error", color = 'dodgerblue')
plt.title("Relu NN: Errors vs. d1 for real data")
plt.legend(loc = "upper right")
plt.show()