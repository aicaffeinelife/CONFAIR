import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from common import *
from dataclasses import dataclass

"""
Test the feature set hypothesis on the german 
credit dataset.
"""


@dataclass
class DatasetObj:
    data: np.ndarray
    target: np.ndarray
    z: np.ndarray = None


def load_german(file="datasets/german_numerical-binsensitive.csv"):
    data = pd.read_csv(file)
    scaler = StandardScaler()
    columns = list(data.columns)
    lbl_idx = columns.index("credit")  # the label index
    data_arr = data.to_numpy()
    y = data_arr[:, lbl_idx]
    X = np.delete(data_arr, lbl_idx, 1)
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, random_state=0, test_size=0.2
    )
    data_train = DatasetObj(X_train, y_train)
    data_test = DatasetObj(X_test, y_test)
    columns.remove("credit")
    return data_train, data_test, columns


def main():
    print("Loading data....")
    train_data, test_data, labels = load_german()
    print("Finished loading...")
    print(f"Train size: {train_data.data.shape}")
    print(f"Test size: {test_data.data.shape}")
    print(f"Labels: {labels}")
    clf = train_svm(train_data.data, train_data.target)
    err_baseline = test_svm(clf, test_data.data, test_data.target)
    _, perm_ids = get_feature_importance(
        clf,
        test_data.data,
        test_data.target,
        labels,
        save="feature_importance_german.png",
    )
    plot_change_in_err(
        clf,
        (train_data.data, train_data.target, test_data.data, test_data.target),
        perm_ids,
        err_baseline,
        save="delta_err_german",
    )


if __name__ == "__main__":
    main()
