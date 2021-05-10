import numpy as np
import csv

def import_train_data(train_X_path,train_Y_path):
    train_X = np.genfromtxt(train_X_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_path, delimiter=',', dtype=np.float64, skip_header=0)
    train_Y = train_Y.reshape((len(train_Y),1))
    train_X = np.hstack((np.ones((len(train_X),1),dtype=np.float64),train_X))
    return train_X,train_Y

def compute_hypothesis(X,W):
    H = np.matmul(X,W)
    return H

def compute_cost(X,Y,W):
    m = len(X)
    H = compute_hypothesis(X,W)
    J = 1/(2*m)*np.sum((H-Y)**2,dtype=np.float64)
    return J

def gradient_descent(W,alpha,X,Y):
    m = len(X)
    J_prev=compute_cost(X,Y,W)
    i=0
    while True:
        H = compute_hypothesis(X,W)
        W = W-(np.matmul((H-Y).T,X).T)/m*alpha
        J_new = compute_cost(X,Y,W)
        if(i%10000==0):
                print(J_new, abs(J_new-J_prev))
        if(abs(J_new-J_prev)<0.000001): break
        else: J_prev = J_new
        i+=1
       
    return W
def train_save_model():
    train_X,train_Y = import_train_data("train_X_lr.csv","train_Y_lr.csv")
    W = gradient_descent(np.array([[0],[0],[0],[0],[0]]),0.0002,train_X,train_Y)
    with open("WEIGHTS_FILE.csv",'w+',newline='') as file:
        wr=csv.writer(file)
        wr.writerows(W)
        file.close()

if __name__=="__main__":
    train_save_model()