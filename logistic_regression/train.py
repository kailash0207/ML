import numpy as np
import csv
import concurrent.futures

def import_training_data(train_X_file_path, train_y_file_path):
    train_X = np.genfromtxt('train_X_lg_v2.csv',delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt('train_Y_lg_v2.csv',delimiter=',', dtype=np.float64).reshape((len(train_X),1))
    train_X = np.hstack((np.ones((len(train_X), 1)),train_X))
    return train_X,train_Y

def hypothesis_fn(X, W):
    h=np.matmul(X,W)
    return h

def sigmoid(Z):
    S=1.0/(1.0+np.exp(-Z, dtype=np.float64))
    return S

def cost_fn(X, Y, W):
    m=len(X)
    Z=hypothesis_fn(X, W)
    A=sigmoid(Z)
    A[A == 1]=0.999999
    A[A == 0]=0.000001
    cost=-1/m*(np.dot(Y.T,np.log(A))+np.dot((1-Y).T,np.log(1-A)))
    return cost

def gradient(X, Y, W):
    m=len(X)
    Z=hypothesis_fn(X, W)
    A=sigmoid(Z)
    A[A == 1]=0.999999
    A[A == 0]=0.000001
    dW=1/m*(np.dot(X.T,A-Y))
    return dW

def modify_labels_to_binary(Y, label):
   train_Y=np.copy(Y)
   train_Y=np.where(train_Y == label, 1, 0)
   return train_Y

def write_to_csv(weights, file="WEIGHTS_FILE.csv"):
    with open(file, 'w', newline='') as csv_file:
        wr=csv.writer(csv_file)
        wr.writerows(weights)
        csv_file.close()

def train_binary_classifier(args):
    X = args[0]
    label = args[2]
    Y = modify_labels_to_binary(args[1],label)
    W = np.zeros((len(X[0]),1),dtype=np.float64)
    lr = 1.0
    is_weights_finalised = False
    while(not is_weights_finalised):
        W_temp = W
        iter_cnt = 0
        while(True):
            iter_cnt+=1
            dW = gradient(X,Y,W_temp)
            W_temp-=(lr*dW)
            cost=cost_fn(X,Y,W_temp)
            if iter_cnt==1: prev_cost=cost
            else:
                if cost-prev_cost>=0:
                    lr/=1.1
                    break
                else:
                    if prev_cost-cost<0.000001:
                        W = W_temp
                        is_weights_finalised = True
                        break
                    else: prev_cost=cost
    
    return label,W.reshape(1,len(W))

    
def train_binary_classifiers_for_each_class(train_X,train_Y):
    weights_for_each_class = [0,0,0,0]
    args = [[train_X,train_Y,label] for label in range(4)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.map(train_binary_classifier,args)
        for label,result in future:
            weights_for_each_class[label] = result

        # for label in range(4):
        #     weights_for_current_class = train_binary_classifier(train_X,train_Y,label)
        #     weights_for_each_class.append(weights_for_current_class)
    #print(weights_for_each_class)
    return np.array(weights_for_each_class, dtype=np.float64).reshape(4,len(weights_for_each_class[0][0]))
if __name__ == "__main__":
    train_X, train_Y = import_training_data("train_X_lg_v2.csv", "train_Y_lg_v2.csv")
    weights_for_each_classifier = train_binary_classifiers_for_each_class(train_X,train_Y) 

    write_to_csv(weights_for_each_classifier);


    
