import numpy as np
import csv
import sys
from validate import validate
from train import classify_points_using_knn


def import_data_and_model_parameters(test_X_file_path, parameters_file_path):
    test_X = np.genfromtxt(
        test_X_file_path, delimiter=",", dtype=np.float64, skip_header=1
    )
    model_parameters = np.genfromtxt(
        parameters_file_path, delimiter=",", dtype=np.int32, skip_header=1
    )
    return test_X, model_parameters


def predict_target_values(test_X, model_parameters):

    train_X = np.genfromtxt(
        "train_X_knn.csv", delimiter=",", dtype=np.float64, skip_header=1
    )
    train_Y = np.genfromtxt("train_Y_knn.csv", delimiter=",", dtype=np.float64)
    pred_Y = classify_points_using_knn(
        train_X, train_Y, test_X, model_parameters[0], model_parameters[1]
    )
    pred_Y = np.array(pred_Y).reshape((len(pred_Y), 1))
    return pred_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    with open(predicted_Y_file_name, "w", newline="") as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path, parameters_file_path):
    test_X, model_parameters = import_data_and_model_parameters(
        test_X_file_path, parameters_file_path
    )
    pred_Y = predict_target_values(test_X, model_parameters)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path, parameters_file_path="model_parameters.csv")

    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv")
