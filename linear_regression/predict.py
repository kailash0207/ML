import numpy as np
import csv
import sys

from validate import validate

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights


def predict_target_values(test_X, weights):

    new=np.ones((len(test_X),1))
    test_X=np.hstack((new,test_X))
    predicted_values=np.matmul(test_X,weights)
    return predicted_values

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w+', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

"""       
def generate_weights_file(test_X_file_path,actual_test_Y_file_path):
    X=np.genfromtxt(test_X_file_path,delimiter=',',dtype=np.float64,skip_header=1)
    new=np.ones((len(X),1))
    X=np.hstack((new,X))
    
    Y=np.genfromtxt(actual_test_Y_file_path,delimiter=',',dtype=np.float64)
    compute=np.linalg.inv(np.matmul(np.transpose(X),X))
    compute2=np.matmul(compute,np.transpose(X))
    final=np.matmul(compute2,Y)
    final=final.reshape(len(final),1)
    with open("WEIGHTS_FILE.csv",'w+',newline='') as file:
        wr=csv.writer(file)
        wr.writerows(final)
        file.close()
"""   

def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_lr.csv")


if __name__ == "__main__":
    test_X_file_path =sys.argv[1];
    
    #generate_weights_file(test_X_file_path, actual_test_Y_file_path="train_Y_lr.csv")

    predict(test_X_file_path)
    
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_lr.csv") 
