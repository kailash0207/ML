import numpy as np
import csv
import preprocessing as p
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
    weights=weights.reshape(1,len(weights))
    with open(file, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(weights)
        csv_file.close()
        
if __name__=="__main__":
        train_X=np.genfromtxt('train_X_pr.csv',delimiter=',',dtype=np.float64,skip_header=1)
        train_Y=np.genfromtxt('train_Y_pr.csv',delimiter=',',dtype=np.float64)
        X=p.feature_scaling(p.replace_null(train_X))
        Y=train_Y
    
        final_W=np.zeros(len(X[0]),dtype=np.float64)
        final_b=0
        A=False
        learning_rate=0.75
        while(A!=True):
            W=np.zeros(len(X[0]),dtype=np.float64)
            b=0
            iter_no=0
            
            while(True):
                iter_no=iter_no+1
                dW,db=gradient(X,Y,W,b)
                W=W-(learning_rate*dW)
                b=b-(learning_rate*db)
                cost=cost_fn(X,Y,W,b)
                if iter_no==1: prev_cost=cost
                else:
                    if cost-prev_cost>=0:
                        learning_rate=learning_rate/1.65
                        break
                    else:
                        if abs(cost-prev_cost)<0.000001:
                            
                            A=True
                            final_W=W
                            final_b=b
                            break
                        else: prev_cost=cost
        
        weights=np.hstack((final_W,final_b))
        print(weights)
        write_to_csv(weights)
    

