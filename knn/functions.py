import math as m
import predict as pdt
import numpy as np
train_X=np.genfromtxt("train_X_knn.csv", delimiter=',', dtype=np.float64, skip_header=1)
train_Y=np.genfromtxt("train_Y_knn.csv", delimiter=',', dtype=np.float64)
def compute_ln_norm_distance(vector1, vector2, n):
    
    S=0
    for i in range(len(vector1)):
        S=S+(abs(vector1[i]-vector2[i]))**n
    return S**(1/n)

def find_k_nearest_neighbors(train_X, test_example, k, n):
    
    distances=[]
    for i in train_X:
        d=compute_ln_norm_distance(i,test_example,n)
        distances.append(d)
    new_d=sorted(distances)
    X=[]
    for i in range(k):
        X.append(distances.index(new_d[i]))

    return X

def calculate_accuracy(predicted_Y, actual_Y):
    
    c=0
    for i in range(len(predicted_Y)):
        if predicted_Y[i]==actual_Y[i]: c=c+1
    accuracy=c/len(predicted_Y)
    return accuracy

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    
    classes=[]
    for i in test_X:
        L=find_k_nearest_neighbors(train_X,i,k,n)
        Z=[]
        for j in L:
            Z.append(train_Y[j])
        Z=sorted(Z)
        
        set_z=list(set(Z))
        
        Max=Z.count(set_z[0])
        
        p=set_z[0]
        for x in set_z:
            if Z.count(x)>Max: 
                Max=Z.count(x)
                p=x
        classes.append(p)
    return classes

def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent=15,n=2):
    
    x=m.floor((100-validation_split_percent)/100*len(train_X))
    validation_set=train_X[x:]
    validation_Y=train_Y[x:]
    accuracy_set=[]
    k=1
    while(k<=x):
        L=classify_points_using_knn(train_X[:x],train_Y[:x],validation_set,k,n)
        accuracy=calculate_accuracy(L,validation_Y)
        accuracy_set.append(accuracy)
        k=k+1
    return (accuracy_set.index(max(accuracy_set))+1)


k=get_best_k_using_validation_set(train_X,train_Y)

