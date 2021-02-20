import numpy as np
import csv
import sys

from validate import validate
import functions as f

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""
def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X


def predict_target_values(test_X):
    
    train_X=import_data("train_X_knn.csv")
    train_Y=np.genfromtxt("train_Y_knn.csv",delimiter=',',dtype=np.float64)
    k=f.k
    n=2
    pred_Y=f.classify_points_using_knn(train_X,train_Y,test_X,k,n)
    pred_Y=np.array(pred_Y)
    return pred_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") 
