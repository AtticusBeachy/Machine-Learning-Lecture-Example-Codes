import numpy as np
import pandas as pd
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
        x_train = lhs(Ndim, samples=Ntrain, criterion="maximin", iterations=20_000)#100_000)#20)
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
        x_test = lhs(Ndim, samples=Ntest, criterion="maximin", iterations=1_000)#20000)#20)
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
# GAUSSIAN PROCESS REGRESSION

tic = time.perf_counter()

# construct gpr model and train on data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

GPR = GaussianProcessRegressor( # kernel = Matern()
        kernel=1.0*RBF(1.0) + WhiteKernel(noise_level=1e-10), alpha=0, optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=25) #250) #

GPR.fit(x_train, y_train)


y_pred, sig_pred = GPR.predict(x_test, return_std=True)
y_pred = y_pred.reshape(-1, 1)
sig_pred = sig_pred.reshape(-1, 1)

toc = time.perf_counter()

###########################################################################
# ACCURACY

def RMSE(y_pred, y_true):
    """
    root mean squared error
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    rmse = np.sqrt(np.mean((y_pred-y_true)**2))
    return(rmse)

def NRMSE(y_pred, y_true):
    """
    Additive normalized root mean squared error
    Use when modeling variables that have additive effects
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    y_mean = np.mean(y_true)
    additive_NRMSE = np.sqrt(np.sum((y_pred-y_true)**2)/np.sum((y_mean-y_true)**2))
    return(additive_NRMSE)


nrmse = NRMSE(y_pred, y_test)
print("nrmse: ", nrmse)

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
ax.errorbar(y_test.flatten(), y_pred.flatten(), yerr=1*sig_pred.flatten(), fmt = "none", color = 'k', marker=None, capsize = 4, elinewidth=0.1, capthick=0.2, alpha = 1.0, label="$\pm 1\sigma$", zorder=1)
ax.scatter(y_test, y_pred, c='g', marker='o', linewidths=0.1, edgecolors='k', label="data")
ax.plot(y_test, y_test, "k-", linewidth=2, label = "true")
ax.legend(loc="upper left")
ax.set_title("GPR (NRMSE = "+string(nrmse)+")")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
plt.savefig(SAVE_PATH+"actual_vs_predicted_gaussian_process_regression.png", format="png", dpi=300, bbox_inches="tight")
plt.close(fig)

###########################################################################
# LOG LIKELIHOOD OF GAUSSIAN PROCESS REGRESSION PREDICTIONS

from scipy.stats import norm

def sum_log_likelihood(y_true, y_pred, scale):
    """
    returns the sum of log likelihoods for a normal distribution
    an error metric that takes model uncertainty or PDF into account
    """
    SLL = np.sum(norm.logpdf(y_true, loc=y_pred, scale=scale))
    return(SLL)


SLL = sum_log_likelihood(y_test, y_pred, sig_pred)
print("sum_log_likelihoods: ", SLL)


print("Time for GPR: ", toc-tic)
