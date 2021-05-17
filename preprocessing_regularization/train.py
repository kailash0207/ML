import numpy as np
import math
import csv
from preprocessing import preprocess_train_data, preprocess_test_data
from sklearn.metrics import f1_score


def hypothesis_fn(X,W,b):
    
    X=np.transpose(X)
    h=np.matmul(W,X)+b
    return h

def sigmoid(Z):
    S=1.0/(1.0+np.exp(-Z))
    return S

def cost_fn(X,Y,W,b,l=1):
    m=len(X)
    Z=hypothesis_fn(X,W,b)
    A=sigmoid(Z)
    A[A==1]=0.999999
    A[A==0]=0.000001
    cost=-1/m*(np.dot(np.log(A),Y)+np.dot(np.log(1-A),1-Y))+l/(2*m)*np.sum(W**2)
    return cost

def gradient(X,Y,W,b,l=1):
    m=len(X)
    Z=hypothesis_fn(X,W,b)
    A=sigmoid(Z)
    A[A==1]=0.999999
    A[A==0]=0.000001
    
    Y=np.transpose(Y)
    dW=1/m*(np.dot(A-Y,X)+l*W)
    db=1/m*np.sum(A-Y)

    return dW,db

def write_to_csv(weights, file="WEIGHTS_FILE.csv"):
    
    with open(file, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(weights)
        csv_file.close()

def binary_classifier(X,Y,learning_rate=1.0,l=1.0,iter_limit=100000):
    W=np.zeros((1,len(X[0])),dtype=np.float64)
    b=0
    iter_no=0
    prev_cost = cost_fn(X,Y,W,b,l)
    while(iter_no<iter_limit):
        iter_no=iter_no+1
        dW,db=gradient(X,Y,W,b,l)
        W-=(learning_rate*dW)
        b-=(learning_rate*db)
        new_cost=cost_fn(X,Y,W,b,l)
        if abs(new_cost-prev_cost)<0.000001: break
        else: prev_cost=new_cost
    weights = np.hstack((W,np.array([b]).reshape(1,1)))
    
    return weights

def train_validate_spit(X,Y,validation_percent):
    a = math.floor((100-validation_percent)/100.0*len(X))
    train_X = X[:a+1,:]
    train_Y = Y[:a+1,:]
    validation_X = X[a+1:,:]
    validation_Y = Y[a+1:,:]
    return train_X,train_Y,validation_X,validation_Y

def predict_target(test_X,weights):
        test_X=preprocess_test_data(test_X)
        Z=hypothesis_fn(test_X,weights[:,:len(weights[0])-1],weights[0][-1])
        A=sigmoid(Z).reshape(len(test_X),1)
        L=np.where(A>=0.5,1,0)
        return L


def tune_hyperparameters(X,Y):
    train_X,train_Y,validation_X,validation_Y = train_validate_spit(X,Y,25)
    train_X=preprocess_train_data(train_X)
    
    learning_rates = [0.01,0.03,0.1,0.3,1]
    lambdas = [1,2.5,5,7.5,10]
    best_lr=0.0001
    best_l = 0.1
    best_f1_score =0
    for l in lambdas:
        for lr in learning_rates:
            weights = binary_classifier(train_X,train_Y,lr,l)
            pred_Y = predict_target(validation_X,weights)
            weighted_f1_score = f1_score(validation_Y, pred_Y, average = 'weighted')
            if(weighted_f1_score>best_f1_score):
                best_f1_score = weighted_f1_score
                best_lr = lr
                best_l = l
    print(best_lr,best_l, best_f1_score)
    return best_l,best_lr


        
if __name__=="__main__":
        
        X=np.genfromtxt('train_X_pr.csv',delimiter=',',dtype=np.float64,skip_header=1)
        Y=np.genfromtxt('train_Y_pr.csv',delimiter=',',dtype=np.float64).reshape(len(X),1)
        l,lr = tune_hyperparameters(X,Y)
        X = preprocess_train_data(X)
        weights = binary_classifier(X,Y,lr,l)
        
        write_to_csv(weights)
    
        
        
       
        
    

