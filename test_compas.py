# from fairness_superfolder.experimental_code.common import plot_change_in_err, plot_feature_importance, test_svm
from typing import Tuple, List
import pandas as pd
import numpy as np
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from common import *


@dataclass
class DatasetObj:
    data: np.ndarray
    target: np.ndarray
    s: np.ndarray = None


def load_compas(
    COMPAS_FILE: str = "datasets/compas-scores-two-years.csv",
) -> Tuple[DatasetObj, DatasetObj, List[str]]:
    FEATURES_CLASSIFICATION = [
        "age_cat",
        "race",
        "sex",
        "priors_count",
        "c_charge_degree",
    ]
    # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CONT_VARIABLES = ["priors_count"]
    CLASS_FEATURE = "two_year_recid"  # the decision variable
    SENSITIVE_ATTRS = ["race"]

    labels = []

    df = pd.read_csv(COMPAS_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"])
    data = df.to_dict("list")
    for k in data.keys():
        data[k] = np.array(data[k])

    idx = np.logical_and(
        data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30
    )
    idx = np.logical_and(idx, data["is_recid"] != -1)
    idx = np.logical_and(idx, data["c_charge_degree"] != "O")
    # idx = np.logical_and(idx, data["c_charge_degree"] != "O")
    idx = np.logical_and(
        idx,
        np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"),
    )

    for k in data.keys():
        data[k] = data[k][idx]
    subset = df.from_dict(data)

    y = data[CLASS_FEATURE]
    y[y == 0] = -1  # label in {-1, 1}
    sens_feats = None
    X = np.array([]).reshape(len(y), 0)
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            labels += [attr]
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)
            vals = np.reshape(vals, (len(y), 1))
        else:
            unq = list(subset[attr].unique())
            if len(unq) > 2:
                lbl = [f"{attr}_" + str(v) for v in unq]
                labels += lbl
            else:
                labels += [attr]
            lb = LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)
            # print(attr, unq, vals.shape)

        # if attr in SENSITIVE_ATTRS:
        #     sens_feats = vals.flatten()

        X = np.hstack([X, vals])

    assert X.shape[1] > 0, f"Invalid shape for data {X.shape}"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    data_train = DatasetObj(X_train, y_train)
    data_test = DatasetObj(X_test, y_test)

    return data_train, data_test, labels


def main():
    train_data, test_data, label_names = load_compas()
    print(f"Train data shape: {train_data.data.shape}")
    print(f"Test data shape: {test_data.data.shape}")
    clf = train_svm(train_data.data, train_data.target)
    err_baseline = test_svm(clf, test_data.data, test_data.target)
    _, perm_ids = get_feature_importance(
        clf,
        test_data.data,
        test_data.target,
        label_names,
        save="figs/compas_features.png",
    )
    plot_change_in_err(
        clf,
        (train_data.data, train_data.target, test_data.data, test_data.target),
        perm_ids,
        err_baseline,
        save="delta_err_propublica",
    )


if __name__ == "__main__":
    main()
