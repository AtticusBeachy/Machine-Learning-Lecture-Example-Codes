import numpy as np
import math
import pickle
from pyDOE2 import lhs, ff2n
import time

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)


def rosenbrock(X, c1=100.0, c2=1.0):
    """
    Takes in a 2D matrix of samples and outputs 2D column matrix of
    Rosenbrock function responses
    The function must be 2D or higher
    """
    Y = np.sum(c1*(X[:,1:] - X[:,:-1]**2.0)**2.0 +c2* (1 - X[:,:-1])**2.0, axis=1)
    return(Y)

###########################################################################
# PROBLEM SETUP

Ndim = 10
Ntrain = 2*(Ndim+1)*(Ndim+2)
Ntest = 10_000
F = rosenbrock
lb = -2*np.ones([Ndim]) 
ub = 2*np.ones([Ndim]) 

SAVE_PATH = "./images/"

# create folder
import os
if not os.path.exists("images"):
   os.makedirs("images")

###########################################################################
# GET TRAIN AND TEST DATA


def get_training_doe(Ntrain, Ndim):
    """ Get training data locations x_train """


    filename_train = f"x_train_E2NN_{Ntrain}_{Ndim}D_samples_no_FF"
    
    try: 
        # load data
        with open(filename_train, "rb") as infile:
            x_train = pickle.load(infile)
    except: 
        # generate data
        print("Generating training samples using LHS")
        x_train = lhs(Ndim, samples=Ntrain, criterion="maximin", iterations=1_000)#100_000)#20)
        print("DOE of training points complete")

        # save data
        with open(filename_train, "wb") as outfile:
            pickle.dump(x_train, outfile)

    return(x_train)


def get_test_doe(Ntest, Ndim):
    """ Get test data locations x_test """

    filename_test = f"x_test_E2NN_{Ntest}_{Ndim}D_samples"
    
    try: 
        # load data
        with open(filename_test, "rb") as infile:
            x_test = pickle.load(infile)
    except: 
        # generate data

        # Get samples using LHS
        print("Generating test samples using LHS")
        x_test = lhs(Ndim, samples=Ntest, criterion="maximin", iterations=100)#20000)#20)
        print("DOE of test points complete")

        # save data
        with open(filename_test, "wb") as outfile:
            pickle.dump(x_test, outfile)

    return(x_test)

x_train = get_training_doe(Ntrain, Ndim)
x_train = (ub-lb)*x_train + lb
y_train = F(x_train)
y_train = y_train.reshape(-1, 1)

x_test = get_test_doe(Ntest, Ndim)
x_test = (ub-lb)*x_test + lb
y_test = F(x_test)
y_test = y_test.reshape(-1, 1)

###########################################################################
# SCALE DATA

from sklearn.preprocessing import MinMaxScaler
xscale_obj = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
x_train = xscale_obj.transform(x_train)
x_test = xscale_obj.transform(x_test)


yscale_obj = MinMaxScaler(feature_range=(-1, 1)).fit(y_train)
y_train = yscale_obj.transform(y_train)
y_test = yscale_obj.transform(y_test)

###########################################################################
# ACCURACY FUNCTIONS

def NRMSE(y_pred, y_true):
    """
    Normalized root mean squared error
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    y_mean = np.mean(y_true)
    nrmse = np.sqrt(np.sum((y_pred-y_true)**2)/np.sum((y_mean-y_true)**2))
    return(nrmse)

def RMSE(y_pred, y_true):
    """
    root mean squared error
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    rmse = np.sqrt(np.mean((y_pred-y_true)**2))
    return(rmse)

###########################################################################
# NEURAL NETWORK K-FOLD CROSS VALIDATION

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

# # Use GPU
# physical_devices = tf.config.list_physical_devices("GPU")
# print("physical devices: ", physical_devices)
# # Don't crash if something else is also using the GPU
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_nn(n_neurons_l1, n_neurons_l2, Ndim, reg):
    """ Get NN with specified architecture and regularization """
    x_input = keras.Input(shape=Ndim)
    L1 = keras.layers.Dense(n_neurons_l1, input_shape = [Ndim],
                            activation="swish",
                            kernel_initializer="glorot_normal",
                            bias_initializer="he_uniform",
                            kernel_regularizer=reg)(x_input)
    L2 = keras.layers.Dense(n_neurons_l2, input_shape = [n_neurons_l1],
                            activation="swish",
                            kernel_initializer="glorot_normal",
                            bias_initializer="he_uniform",
                            kernel_regularizer=reg)(L1)
    NN_output = keras.layers.Dense(1, input_shape=[n_neurons_l2],
                                     activation = "linear")(L2)
    NN_model = keras.Model(
                           inputs=x_input,
                           outputs = NN_output
                           )
    return(NN_model)


def train_nn(NN_model, x_train, y_train, x_test, y_test, n_epoch, tol, max_epoch):
    """
    Train given nn according to specifications
    n_epoch: number of epochs to run before checking improvement
    tol: Tolerance of improvement. Converge if improvement below tol
    """

    last_best_loss = np.inf

    losses = np.array([])
    validation_losses = np.array([])
    validation_iters_all = np.array([])
    loop_iter = 0
    # loop for training NN
    while True:
        validate_iters = np.arange(1, n_epoch+1, 100) # add 1 because end excluded
        history_NN = NN_model.fit(x_train, y_train,
                                  validation_data = (x_test, y_test),
                                  validation_freq = validate_iters.tolist(),
                                  batch_size = 512, #2048, #256, #32, # default 32
                                  epochs = n_epoch)
        best_loss = np.min(history_NN.history["loss"][-100:])
        if losses.size:
            losses = np.hstack([losses, history_NN.history["loss"]])
            validation_losses = np.hstack([validation_losses, history_NN.history["val_loss"]])
            validation_iters_all = np.hstack([validation_iters_all, validate_iters+loop_iter*n_epoch])
        else:
            losses = np.array(history_NN.history["loss"])
            validation_losses = np.array(history_NN.history["val_loss"])
            validation_iters_all = validate_iters

        improvement = last_best_loss-best_loss
        print("best_loss: ", best_loss)
        print("Improvement: ", improvement)
        last_best_loss = best_loss
        loop_iter += 1
        if improvement < tol or losses.size >= max_epoch:
            break

    return(NN_model, losses, validation_losses, validation_iters_all)




###########################################################################
# NEURAL NETWORK TRAINING AND PREDICTION

opt = keras.optimizers.Adam(learning_rate=0.001) #0.01)# 0.001 default
reg_val = 1e-4 #3e-4 #

n_epoch = 10_000 #20 #5_000 #
tol = 1e-5 #1e-2 #1e-3 #
max_epoch = 200_000

neuron_val = 100

tic = time.perf_counter()

reg = keras.regularizers.l2(reg_val)
n_neurons_l1 = neuron_val
n_neurons_l2 = neuron_val
NN_model = get_nn(n_neurons_l1, n_neurons_l2, Ndim, reg)
NN_model.compile(loss="mse", optimizer=opt)
NN_model, losses, validation_losses, validation_iters_all = train_nn(NN_model, x_train, y_train, x_test, y_test, n_epoch, tol, max_epoch)
# get NN prediction
y_pred = NN_model.predict(x_test)

toc = time.perf_counter()

print("Time for NN: ", toc-tic)

################################################################################
# PLOT OF LOSS HISTORY

fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111)
# ax.plot(history_NN.history["loss"], "r-", label="train loss")
# ax.plot(validate_iters, history_NN.history["val_loss"], "b-", label="test loss")
ax.plot(losses, "r-", label="train loss")
ax.plot(validation_iters_all, validation_losses, "b-", label="test loss")
ax.set_title("NN loss history")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.legend(loc="upper right")
plt.semilogy()
plt.savefig(SAVE_PATH+"loss_history_nn_get_neuron.png", format="png", dpi=300)
plt.close(fig)

###########################################################################
# ACCURACY

nrmse = NRMSE(y_pred, y_test)
print("\n")
print("nrmse: \n", nrmse)
rmse = RMSE(y_pred, y_test)
print("rmse: \n", rmse)

###########################################################################
# PLOT ACTUAL VS PREDICTED

def string(val, dig=4):
    """ For rounding error measures when included in plot titles """
    # val = np.round(val, dig)
    # s = str(val)
    if val < 0.01:
        s = np.format_float_scientific(val, precision=dig)
    else:
        s = np.format_float_positional(val, precision=dig)
    return(s)

fig = plt.figure(figsize=[6.4, 4.8])
ax = fig.add_subplot(111)
ax.scatter(y_test, y_pred, c='g', marker='o', linewidths=0.1, edgecolors='k', label="data")
ax.plot(y_test, y_test, "k-", linewidth=2, label = "true")
ax.legend(loc="upper left")
ax.set_title("Neural Network (NRMSE = "+string(nrmse)+")")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
plt.savefig(SAVE_PATH+"actual_vs_predicted_neural_network_get_neuron.png", format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

