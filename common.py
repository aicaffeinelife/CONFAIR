import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

param_grid = [
    {
        "C": list(np.logspace(-4, 4, num=30)),
        "gamma": [0.001, 0.01, 0.1, 1],
        "kernel": ["rbf"],
    }
]


def train_svm(train_data, train_label, param_grid=param_grid):
    """train svm"""
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid=param_grid, n_jobs=8)
    clf.fit(train_data, train_label)
    print(f"Best estimator: {clf.best_estimator_}")
    return clf


def test_svm(model, test_data, test_label):
    """test svm and measure error rate"""
    pred = model.predict(test_data)
    acc = accuracy_score(test_label, pred)
    err = 1.0 - acc
    print(f"Error rate: {err:.3f}")
    return err


def get_feature_importance(
    model,
    test_data,
    test_label,
    label_list=None,
    plot=False,
    save="feature_importance.png",
):
    """Given a fitted model plot the importance of features and save them to file
    and return the importance object"""
    r = permutation_importance(
        model, test_data, test_label, n_repeats=30, random_state=0
    )
    # print(r.importances)

    if isinstance(label_list, list):
        label_list = np.array(label_list)
    perm_ids = r.importances_mean.argsort()[::-1]  # ordered from most to least imp
    if label_list is not None:
        for i in perm_ids:
            print(f"Feature: {label_list[i]}:  {r.importances_mean[i]: .3f}")
    if plot:
        plt.boxplot(
            r.importances[perm_ids].T,
            vert=False,
            labels=label_list[perm_ids],
            showfliers=False,
        )
        if save:
            plt.savefig(f"figs/{save}")
    return r, perm_ids


def plot_change_in_err(
    model, data, perm_ids, err_baseline, k=5, save="delta_err_generic", **kwargs
):
    """Given a model, compute the change in error rate and plot it"""
    train_data, train_label, test_data, test_label = data
    cases = ["top-k", "bottom-k"]
    pgrid = kwargs.get("param_grid", None)
    print(f"Fitting on Top {k} features")
    tr_data, te_data = (
        np.copy(train_data)[:, perm_ids[:k]],
        np.copy(test_data)[:, perm_ids[:k]],
    )
    model = (
        train_svm(tr_data, train_label, param_grid=pgrid)
        if pgrid
        else train_svm(tr_data, train_label)
    )
    err_topk = test_svm(model, te_data, test_label)
    print(f"Top k error: {err_topk:.3f}")
    print(f"Fitting on Bottom {k} features")
    tr_data_1, te_data_1 = (
        np.copy(train_data)[:, perm_ids[::-1][:k]],
        np.copy(test_data)[:, perm_ids[::-1][:k]],
    )
    model = (
        train_svm(tr_data_1, train_label, param_grid=pgrid)
        if pgrid
        else train_svm(tr_data_1, train_label)
    )
    err_botk = test_svm(model, te_data_1, test_label)
    errs = [abs(err_baseline - err_topk), abs(err_baseline - err_botk)]
    # plotting
    width = 0.25
    x = np.arange(len(cases))
    plt.bar(x, errs, width, alpha=0.5)
    plt.xlabel(cases)
    plt.ylabel("Delta Error rate")
    if save:
        plt.savefig(f"figs/{save}.png")


# def _plot_barplot(deltas, cases, save='generic_save_dataset'):
#     '''Plot barplot of deltas in a single plot'''
