# from experimental_code.data_utils import load_bank_data
import numpy as np
import warnings
import argparse
from typing import Callable, Tuple

from sklearn.utils import shuffle
from data_utils import *
from fair_svm import FairSVM
from fairness_metrics import EO_true_positive, neg_predict_value, treatment_equality
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn import svm
from sklearn.metrics import accuracy_score

DATASET_MAP = {
    "german": load_german,
    "drug": load_drug_data,
    "compas": load_compas,
    "bank": load_bank_data,
    "arrhythmia": load_arrhythmia_data,
}


def run_optimization(dataset, run_fn, param_grid=None, times=5, linear=False):
    """
    Run optimization and compute stats
    """
    run_acc, run_eq, run_npv = [], [], []
    for i in range(times):
        print(f"Run {i}")
        acc, eq_val, npv_val = run_fn(dataset, param_grid, linear=linear)
        run_acc.append(acc)
        run_eq.append(eq_val)
        run_npv.append(npv_val)

    print(f"Run Acc: {np.mean(run_acc)} ({np.std(run_acc)})")
    print(f"Run DEO: {np.mean(run_eq)} ({np.std(run_eq)})")
    print(f"Run NPV: {np.mean(run_npv)} ({np.std(run_npv)})")


def train_baseline(
    dataset: str, param_grid=None, linear=False
) -> Tuple[float, float, float]:
    if not param_grid:
        warnings.warn(
            "The testing on different folds will be done with default parameters"
        )
    cv = 10  # use on default split
    if dataset == "adult":
        data_train, data_test, sens = load_adult(smaller=True)
    else:
        data, sens = DATASET_MAP[dataset]()
        # cv = RepeatedStratifiedKFold(n_splits=10, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, shuffle=True, test_size=0.2
        )
        data_train = DatasetObj(X_train, y_train)
        data_test = DatasetObj(X_test, y_test)

    sens_arr = data_train.data[:, sens]
    sens_val = list(set(sens_arr))
    print("Sens vals: ", sens_val)
    print("Dataset size: ", data.data.shape)
    model = svm.SVC() if not linear else svm.SVC(kernel="linear")
    # 10 fold cross validation
    clf = (
        GridSearchCV(model, cv=cv, param_grid=param_grid, n_jobs=8)
        if param_grid
        else model
    )
    clf.fit(data_train.data, data_train.target)
    print(f"Best estimator: {clf.best_estimator_}")
    pred_test = clf.predict(data_test.data)
    acc = accuracy_score(data_test.target, pred_test)
    eq_dict = EO_true_positive(clf, data_test, sens)
    npv_dict = neg_predict_value(clf, data_test, sens)
    eq_val = np.abs(eq_dict[sens_val[0]] - eq_dict[sens_val[1]])
    npv_val = np.abs(npv_dict[sens_val[0]] - npv_dict[sens_val[1]])
    print("Dataset: ", dataset)
    print("Accuracy: ", acc)
    print("DEO of Baseline: ", np.abs(eq_dict[sens_val[0]] - eq_dict[sens_val[1]]))
    print("NPV of Baseline: ", np.abs(npv_dict[sens_val[0]] - npv_dict[sens_val[1]]))
    return acc, eq_val, npv_val


def train_fairsvm(
    dataset: str, param_grid=None, ftol=1e-2, sense="deo", linear=False
) -> None:
    if not param_grid:
        warnings.warn(
            "The testing on different folds will be done with default parameters"
        )
    cv = 10  # use on default split
    if dataset == "adult":
        data_train, data_test, sens = load_adult(smaller=True)
    else:
        data, sens = DATASET_MAP[dataset]()
        # cv = RepeatedStratifiedKFold(n_splits=10, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, shuffle=True, test_size=0.2
        )
        data_train = DatasetObj(X_train, y_train)
        data_test = DatasetObj(X_test, y_test)

    sens_arr = data_train.data[:, sens]
    sens_val = list(set(sens_arr))
    print("Sens vals: ", sens_val)
    print("Dataset size: ", data_train.data.shape)
    model = (
        FairSVM(sens_feat=sens, ftol=ftol)
        if not linear
        else FairSVM(sens_feat=sens, kernel="linear", ftol=ftol)
    )
    clf = (
        GridSearchCV(model, cv=cv, param_grid=param_grid, n_jobs=8)
        if param_grid
        else model
    )
    clf.fit(data_train.data, data_train.target)
    print(f"Best estimator: {clf.best_estimator_}")
    pred_test = clf.predict(data_test.data)
    acc = accuracy_score(data_test.target, pred_test)
    eq_dict = (
        EO_true_positive(clf, data_test, sens)
        if sense == "deo"
        else EQ_true_negative(clf, data_test, sens)
    )
    npv_dict = neg_predict_value(clf, data_test, sens)
    eq_val = np.abs(eq_dict[sens_val[0]] - eq_dict[sens_val[1]])
    npv_val = np.abs(npv_dict[sens_val[0]] - npv_dict[sens_val[1]])
    print("Dataset: ", dataset)
    print("Accuracy: ", acc)
    print("DEO of FairSVM: ", np.abs(eq_dict[sens_val[0]] - eq_dict[sens_val[1]]))
    print("NPV of FairSVM: ", np.abs(npv_dict[sens_val[0]] - npv_dict[sens_val[1]]))
    return acc, eq_val, npv_val


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="german")
    parser.add_argument("--ftol", type=float, default=1e-2)
    parser.add_argument("--baseline", action="store_true", help="Run baseline")
    parser.add_argument("--linear", action="store_true", help="Run Linear (Fair)SVM")
    args = parser.parse_args()

    if args.linear:
        param_grid = [{"C": list(np.logspace(-4, 4, num=30))}]
    else:
        param_grid = [
            {
                "C": list(np.logspace(-4, 4, num=30)),
                "gamma": [0.001, 0.01, 0.1, 1],
                "kernel": ["rbf"],
            }
        ]

    is_linear = args.linear
    if args.baseline:
        print("Training baseline...")
        run_optimization(
            dataset=args.dataset,
            run_fn=train_baseline,
            param_grid=param_grid,
            linear=is_linear,
        )
        # train_baseline(dataset=args.dataset, param_grid=param_grid)
    else:
        print("Training FairSVM...")
        run_accs, run_eqs, run_npvs = [], [], []
        for i in range(5):
            print("Run: ", i)
            acc, eqv, npv = train_fairsvm(
                dataset=args.dataset,
                param_grid=param_grid,
                ftol=args.ftol,
                sense="deo",
                linear=is_linear,
            )
            run_accs.append(acc)
            run_eqs.append(eqv)
            run_npvs.append(npv)
        print(f"FairSVM Run Acc: {np.mean(run_accs)} ({np.std(run_accs)})")
        print(f"FairSVM Run DEO: {np.mean(run_eqs)} ({np.std(run_eqs)})")
        print(f"FairSVM Run NPV: {np.mean(run_npvs)} ({np.std(run_npvs)})")

        # run_optimization(
        #     dataset=args.dataset,
        #     run_fn=train_fairsvm,
        #     param_grid=param_grid,
        #     ftol=args.ftol,
        #     sense="deo",
        # )
        # train_fairsvm(dataset=args.dataset, k=10, ftol=args.ftol, param_grid=param_grid)

    # data_train, data_test, sens = load_adult(smaller=True)
    # data_train, data_test, sens = load_german()
    # sens_arr = data_train.data[:, sens]
    # sens_vals = list(set(sens_arr))
    # print("Sens vals: ", sens_vals)
    # print("Train dataset size: ", data_train.data.shape)
    # print("Test dataset size: ", data_test.data.shape)
    # print("Grid searching for vanilla SVM...")
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, param_grid=param_grid, n_jobs=8)
    # clf.fit(data_train.data, data_train.target)
    # print(f"SVM Best Estimator: {clf.best_estimator_}")

    # pred_test = clf.predict(data_test.data)
    # acc = accuracy_score(data_test.target, pred_test)
    # print(f"SVM accuracy: {acc}")

    # eo_dict = EO_true_positive(clf, data_test, sens)
    # npv_dict = neg_predict_value(clf, data_test, sens)
    # te_dict = treatment_equality(clf, data_test, sens)
    # print(f"DEO of SVM: {np.abs(eo_dict[sens_vals[0]] - eo_dict[sens_vals[1]])}")
    # print(f"NPV of SVM: {np.abs(npv_dict[sens_vals[0]] - npv_dict[sens_vals[1]])}")
    # print(
    #     f"Treatment Equality of SVM {np.abs(te_dict[sens_vals[0]] - te_dict[sens_vals[1]])}"
    # )

    # print("Grid searching for our improvement....")
    # clf = GridSearchCV(fsvm, param_grid=param_grid, n_jobs=8)
    # clf.fit(data_train.data, data_train.target)
    # print(f"Fair SVM Best Estimator: {clf.best_estimator_}")
    # # print("Gamma: ", clf.gamma)

    # pred_test = clf.predict(data_test.data)
    # acc = accuracy_score(data_test.target, pred_test)
    # print("Accuracy of fsvm: ", acc)

    # eo_dict = EO_true_positive(clf, data_test, sens)
    # npv_dict = neg_predict_value(clf, data_test, sens)
    # te_dict = treatment_equality(clf, data_test, sens)

    # print(f"DEO of FairSVM: {np.abs(eo_dict[sens_vals[0]] - eo_dict[sens_vals[1]])}")
    # print(f"NPV of FairSVM: {np.abs(npv_dict[sens_vals[0]] - npv_dict[sens_vals[1]])}")
    # print(
    #     f"Treatment Equality of FairSVM {np.abs(te_dict[sens_vals[0]] - te_dict[sens_vals[1]])}"
    # )


if __name__ == "__main__":
    main()
