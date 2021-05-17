import numpy as np
from scipy.stats import mode
import csv


def one_hot_encoding(X):
    extra_features = np.zeros((len(X),3))
    for i in range(len(X)):
        extra_features[i,int(X[i,3])]=1
    X = np.hstack((X,extra_features))
    X = np.hstack((X[:,0:3],X[:,4:]))
    return X

def replace_null(X):
    mean_mode=np.genfromtxt("mean_mode.csv",delimiter=',',dtype=np.float64)
    inds=np.where(np.isnan(X))
    X[inds]=np.take(mean_mode,inds[1])
    return X

def find_mean_mode(X):
    col_mean=np.nanmean(X[:,[2,5]],axis=0)
    col_mode = (mode(X[:,[0,1,3,4,6]], nan_policy='omit').mode).flatten()
    mean_mode = [*col_mode[0:2],col_mean[0],*col_mode[2:4],col_mean[1],col_mode[4]]
    with open('mean_mode.csv','w',newline='') as file:
        wr = csv.writer(file)
        wr.writerow(mean_mode)
        file.close()

def find_min_max(X):
    col_min=np.nanmin(X[:,(2,5)],axis=0)
    col_max=np.nanmax(X[:,(2,5)],axis=0)
    with open('min_max.csv','w',newline='') as file:
        wr = csv.writer(file)
        wr.writerow(col_min)
        wr.writerow(col_max)
        file.close()
   


def feature_scaling(X):
    min_max = np.genfromtxt('min_max.csv',delimiter=',', dtype=np.float64)
    X[:,2]=(X[:,2]-min_max[0][0])/(min_max[1][0]-min_max[0][0])
    X[:,4]=(X[:,4]-min_max[0][1])/(min_max[1][1]-min_max[0][1])
    return X

def preprocess_train_data(train_X):
    find_mean_mode(train_X)
    train_X = replace_null(train_X)
    train_X = one_hot_encoding(train_X)
    find_min_max(train_X)
    train_X = feature_scaling(train_X)
    return train_X

def preprocess_test_data(test_X):
    test_X = replace_null(test_X)
    test_X = one_hot_encoding(test_X)
    test_X = feature_scaling(test_X)
    return test_X
    




