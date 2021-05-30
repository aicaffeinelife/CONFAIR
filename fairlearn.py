from numpy.lib.arraysetops import unique
from numpy.lib.function_base import cov
from sklearn.base import BaseEstimator
import argparse

# from fairness_superfolder.experimental_code.data_utils import DatasetObj
from typing import Callable, Any, Iterable, Union, List, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.covariance import EmpiricalCovariance
from data_utils import *
from fair_svm import FairSVM
from sklearn import svm
from fairness_metrics import *
from common import get_feature_importance, train_svm

"""
The FAIRLEARN Procedure for learning 
a fair classifier under given fairness 
constraint and a user specified tolerance 
parameter.
"""
DATASET_MAP = {
    "german": load_german,
    "bank": load_bank_data,
    "compas": load_compas,
}


def fairlearn(
    dataset: str,
    clf: Union[svm.SVC, BaseEstimator],
    feat_select_func: Callable[
        [Union[BaseEstimator, svm.SVC], DatasetObj, DatasetObj],
        Tuple[List[int], List[int]],
    ],
    param_grid=None,
    ftol: float = 0.1,
    k: int = 5,
    linear: bool = False,
) -> BaseEstimator:
    """
    Returns a fairer estimator of data based on
    criteria and sense.
    """
    fclf = None
    if dataset == "adult":
        # TODO: before paper submission make sure to evaluate this on the full
        data_train, data_test, sens = load_adult(smaller=True)
    else:
        data, sens = DATASET_MAP[dataset]()
        # TODO: use K-fold, using a simple test train split for now
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=True, test_size=0.2
        )
        data_train = DatasetObj(X_train, y_train)
        data_test = DatasetObj(X_test, y_test)
    print("Starting critical set finding....")
    crit_feat_idx, cov_idxs = feat_select_func(clf, data_train, data_test, sens)
    overlap = list(set(crit_feat_idx).intersection(set(cov_idxs[1:])))
    print("Covars: ", cov_idxs)
    print("Num Overlap: ", len(overlap))
    # TODO: pass "sense" to FairSVM constructor
    print("Critical Feature Indices: ", crit_feat_idx)
    new_sens_idx = None
    if sens in crit_feat_idx[:k]:
        print("Feature in critical set...")
        new_sens_idx = crit_feat_idx.index(sens)
        fsvm = (
            FairSVM(sens_feat=new_sens_idx, ftol=ftol)
            if not linear
            else FairSVM(sens_feat=new_sens_idx, kernel="linear", ftol=ftol)
        )
        fclf = GridSearchCV(fsvm, param_grid=param_grid, n_jobs=8)
        X = data_train.data[:, crit_feat_idx[:k]]
        y = data_train.target
        fclf.fit(X, y)
        data_train.data = data_train.data[:, crit_feat_idx[:k]]
        data_test.data = data_test.data[:, crit_feat_idx[:k]]  #
        print(f"Best fair estimator: {fclf.best_estimator_}")
    elif len(overlap) > 0:
        print("Sens feature not explicitly in critical set...")
        crit_feats = None
        is_bivalued = len(np.unique(data_train.data[:, cov_idxs[1]])) == 2
        if cov_idxs[1] in crit_feat_idx[:k]:
            crit_feats = (
                crit_feat_idx[:k] if is_bivalued else crit_feat_idx[:k] + [sens]
            )
        else:
            crit_feats = (
                crit_feat_idx[:k] + [cov_idxs[1]]
                if is_bivalued
                else crit_feat_idx[:k] + [cov_idxs[1], sens]
            )
        print("New crit feats: ", crit_feats)
        new_sens_idx = (
            crit_feats.index(cov_idxs[1]) if is_bivalued else crit_feats.index(sens)
        )  # the most co-related feature from the sensitive
        X = data_train.data[
            :, crit_feats
        ]  # the first one is going to be the sens feature itself
        y = data_train.target
        fsvm = (
            FairSVM(sens_feat=new_sens_idx, ftol=ftol)
            if not linear
            else FairSVM(sens_feat=new_sens_idx, kernel="linear", ftol=ftol)
        )
        fclf = GridSearchCV(fsvm, param_grid=param_grid, n_jobs=8)
        fclf.fit(X, y)
        data_train.data = data_train.data[:, crit_feats]
        data_test.data = data_test.data[:, crit_feats]
        print(f"Best fair estimator: {fclf.best_estimator_}")
    else:
        fclf = clf

    pred_test = fclf.predict(data_test.data)
    sens_vals = list(set(data_train.data[:, new_sens_idx]))
    print(new_sens_idx)
    print(sens_vals)
    acc = accuracy_score(data_test.target, pred_test)
    eq_dict = EO_true_positive(fclf, data_test, new_sens_idx)
    npv_dict = neg_predict_value(fclf, data_test, new_sens_idx)
    # print(eq_dict)
    eq_val = (
        np.abs(eq_dict[sens_vals[0]] - eq_dict[sens_vals[1]])
        if len(eq_dict) == 2
        else 0.0
    )
    npv_val = (
        np.abs(npv_dict[sens_vals[0]] - npv_dict[sens_vals[1]])
        if len(npv_dict) == 2
        else 0.0
    )

    return acc, eq_val, npv_val


def estimate_feature_importance(
    model: svm.SVC, train_data: DatasetObj, test_data: DatasetObj, sens_feat: int
) -> Tuple[List[int], List[int]]:
    """
    Find the most important feature for classification
    """
    model = train_svm(train_data.data, train_data.target)
    _, perm_ids = get_feature_importance(model, test_data.data, test_data.target)
    cov_ids = get_feature_covariance(train_data.data, sens_feat)
    perm_ids, cov_ids = list(perm_ids), list(cov_ids)
    return perm_ids, cov_ids


def get_feature_covariance(X: np.ndarray, s: int) -> List[int]:
    """
    Estimate the feature covariance
    from the data and get the indices
    of the features that have the maximum
    covariance with the given sensitive attribute
    """
    covs = EmpiricalCovariance().fit(X)
    covar = covs.covariance_
    cov_ids = covar[:, s].argsort()[::-1]
    tru_covars = [i for i in cov_ids if covar[s, i] > 0.0]
    return tru_covars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ftol", type=float, default=1e-2, help="Fairness tolerance threshold"
    )
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset")
    parser.add_argument("--linear", action="store_true", help="Run Linear FAIRLEARN")
    args = parser.parse_args()

    # data_map = {
    #     "adult": load_adult,
    #     "compas": load_compas,
    #     "german": load_german,
    # }
    # # TODO: make this entire training procedure self-contained. So instead
    # of data map it'll be simply train_map, with right dataset
    # if args.dataset == "adult":
    #     data_train, data_test, sens = data_map[args.dataset](smaller=True)
    # else:
    #     data_train, data_test, sens = data_map[args.dataset]()

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

    svc = svm.SVC()
    run_accs = []
    run_eqs = []
    run_npvs = []

    if is_linear:
        print("Beginning LINEAR FAIRLEARN..")
    else:
        print("Beginning FAIRLEARN..")
    for i in range(5):
        print("Run: ", i)
        acc, eqv, npv = fairlearn(
            args.dataset,
            svc,
            estimate_feature_importance,
            param_grid=param_grid,
            ftol=args.ftol,
            linear=is_linear,
        )
        run_accs.append(acc)
        run_eqs.append(eqv)
        run_npvs.append(npv)

    print(f"Run Accs: {np.mean(run_accs)}({np.std(run_accs)})")
    print(f"Run DEO: {np.mean(run_eqs)}({np.std(run_eqs)})")
    print(f"Run NPV: {np.mean(run_npvs)}({np.std(run_npvs)})")
    # pred_test = clf.predict(data_test.data)
    # sens_vals = list(set(data_train.data[:, sens]))
    # print("Sens vals: ", sens_vals)
    # print("Accuracy of fair classifier: ", accuracy_score(data_test.target, pred_test))
    # if args.sense == "deo":
    #     eo_dict = EO_true_positive(clf, data_test, sens)
    #     print(
    #         "DEO of fair classifier: ",
    #         np.abs(eo_dict[sens_vals[0]] - eo_dict[sens_vals[1]]),
    #     )
    # else:
    #     enpr_dict = EQ_npr(clf, data_test, sens)
    #     print(
    #         "Eq NPR of  fair classifier: ",
    #         np.abs(enpr_dict[sens_vals[0]] - enpr_dict[sens_vals[1]]),
    #     )

    # npv_dict = neg_predict_value(clf, data_test, sens)
    # te_dict = treatment_equality(clf, data_test, sens)
    # print(eo_dict)
    # print(npv_dict)
    # print(te_dict)

    # print(
    #     "NPV of fair classifier: ",
    #     np.abs(npv_dict[sens_vals[0]] - npv_dict[sens_vals[1]]),
    # )
    # print(
    #     "Treat. Eq of  fair classifier: ",
    #     np.abs(te_dict[sens_vals[0]] - te_dict[sens_vals[1]]),
    # )


if __name__ == "__main__":
    main()
