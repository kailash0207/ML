import numpy as np
import csv

def import_train_data(train_X_path,train_Y_path):
    train_X = np.genfromtxt(train_X_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_path, delimiter=',', dtype=np.float64, skip_header=0)
    train_Y = train_Y.reshape((len(train_Y),1))
    train_X = np.hstack((np.ones((len(train_X),1),dtype=np.float64),train_X))
    return train_X,train_Y

def calculate_weights(X,Y):
    temp1=np.linalg.inv(np.matmul(np.transpose(X),X))
    temp2=np.matmul(temp1,np.transpose(X))
    weights=np.matmul(temp2,Y).reshape((len(X[0]),1))
    return weights
    

def save_model():
    train_X,train_Y = import_train_data("train_X_lr.csv","train_Y_lr.csv")
    W = calculate_weights(train_X,train_Y)
    with open("WEIGHTS_FILE.csv",'w+',newline='') as file:
        wr=csv.writer(file)
        wr.writerows(W)
        file.close()

if __name__=="__main__":
    save_model()