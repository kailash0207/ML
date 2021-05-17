import numpy as np
import csv
import sys

from validate import validate

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    test_X = np.hstack((np.ones((len(test_X),1)),test_X))
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights

def sigmoid(Z):
    S=1.0/(1.0+np.exp(-Z))
    return S

def predict_target_values(test_X, weights):
    W=weights
    Z=np.dot(W,np.transpose(test_X))
    A=sigmoid(Z)
    A[A==1] = 0.99999
    A[A==0] = 0.00001
    pred_Y = np.argmax(A,axis=0).reshape((len(test_X),1))
    return pred_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg_v2.csv") 
