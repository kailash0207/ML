import math as m
import numpy as np
import csv

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
    for test_example in test_X:
        nearest_neighbours=find_k_nearest_neighbors(train_X,test_example,k,n)
        neighbour_class=[]
        for j in nearest_neighbours:
            neighbour_class.append(train_Y[j])
        neighbour_class.sort()
        
        uniq_neighbour=list(set(neighbour_class))
        
        cnt=neighbour_class.count(uniq_neighbour[0])
        pred_class=uniq_neighbour[0]
        for x in uniq_neighbour:
            a=neighbour_class.count(x)
            if a>cnt: 
                cnt=a
                pred_class=x
        classes.append(pred_class)
    return classes

def get_best_k_n_using_validation_set(train_X, train_Y, validation_split_percent=20):
    x=m.floor((100-validation_split_percent)/100*len(train_X))
    validation_X,validation_Y = train_X[x:], train_Y[x:]
    best_accuracy=0
    best_k=0
    best_n=0
    for k in range(1,x+1):
        best_accuracy_for_k=0
        best_n_for_k=0
        for n in range(1,6):
            pred_Y=classify_points_using_knn(train_X[:x],train_Y[:x],validation_X,k,n)
            accuracy=calculate_accuracy(pred_Y,validation_Y)
            if(accuracy>best_accuracy_for_k):
                best_accuracy_for_k=accuracy
                best_n_for_k=n
        if(best_accuracy_for_k>best_accuracy):
            best_accuracy = best_accuracy_for_k
            best_n = best_n_for_k
            best_k = k 
    print(best_accuracy)   
    return best_k, best_n

def train_model(train_X,train_Y):
     
    k,n = get_best_k_n_using_validation_set(train_X,train_Y)
    model_parameters = np.array((k,n), dtype = np.int32)
    with open("model_parameters.csv",'w', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(["k","n"])
        wr.writerow(model_parameters)
        file.close()
    
if __name__ == '__main__':

    train_X=np.genfromtxt("train_X_knn.csv", delimiter=',', dtype=np.float64, skip_header=1)
    train_Y=np.genfromtxt("train_Y_knn.csv", delimiter=',', dtype=np.float64) 
    train_model(train_X,train_Y)

