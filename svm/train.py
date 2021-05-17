import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def save_min_max(X):
    col_min=np.min(X,axis=0)
    col_max=np.max(X,axis=0)
    with open("min_max.csv",'w',newline='') as file:
        wr = csv.writer(file)
        wr.writerow(col_min)
        wr.writerow(col_max)
        file.close()

def feature_scaling(X):
    col_min, col_max = np.genfromtxt("min_max.csv",delimiter=',',dtype = np.float64)
    return (X - col_min)/(col_max-col_min)

def train_SVM(train_X,train_Y,kernel,gamma,C):
    model =  SVC(C=C,kernel=kernel,gamma=gamma)
    model.fit(train_X,train_Y)
    return model

def preprocess_train_data(train_X):
    save_min_max(train_X)
    train_X = feature_scaling(train_X)
    return train_X

def tune_hyperparameters(train_X, validation_X, train_Y, validation_Y):
    train_X = preprocess_train_data(train_X)
    kernels = ["rbf","linear"]
    Cs = [10,100,500,1000,10000]
    best_kernel = ""
    best_gamma = 0.0
    best_C = 0
    best_accuracy_score = 0
    for kernel in kernels:
        if kernel == "rbf":
            gammas = [1e-4,1e-3,1e-2]
        else:
            gammas = [1e-4]
        for gamma in gammas:
            for C in Cs:
                model = train_SVM(train_X,train_Y,kernel,gamma,C)
                prediction = model.predict(validation_X)
                accuracy = accuracy_score(validation_Y,prediction)
                if(accuracy>best_accuracy_score):
                    best_accuracy_score = accuracy
                    best_gamma = gamma
                    best_C = C
                    best_kernel = kernel
    
    return best_kernel, best_gamma, best_C

def save_model(model,file_name):
    with open(file_name,'wb') as file:
        pickle.dump(model,file)
        file.close()



if __name__=="__main__":
    X=np.genfromtxt("train_X_svm.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_svm.csv",delimiter=",",dtype=np.float64)
    train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, random_state= 42, test_size=0.30)
    params = tune_hyperparameters(train_X, validation_X, train_Y, validation_Y)
    X = preprocess_train_data(X)
    model = train_SVM(X,Y,params[0],params[1],params[2])
    save_model(model,"model_file.sav")


