from operator import pos
from typing import Dict, Any
import numpy as np
from sklearn.utils.validation import check_X_y
from data_utils import DatasetObj
from sklearn.base import BaseEstimator

"""
Fairness Measures of Classifiers.
"""


def EO_true_positive(
    model: BaseEstimator, data: DatasetObj, sens_feature: int, ylabel: int = 1
) -> Dict[Any, float]:
    """
    Compute the Equal Opportunity metric for a given data and classifier.
    """

    preds = model.predict(data.data)
    gt = data.target
    sens_feature_arr = data.data[:, sens_feature]
    sens_vals = list(set(sens_feature_arr))
    print(np.unique(preds))
    eo_dict = {}
    for val in sens_vals:
        eq_tmp = None
        # pos_sensitive = np.sum(
        #     np.logical_and(data.data[:, sens_feature] == val, gt == ylabel)
        # )
        pos_sensitive = np.sum(
            [
                1.0 if data.data[i, sens_feature] == val and gt[i] == ylabel else 0.0
                for i in range(len(preds))
            ]
        )
        if pos_sensitive > 0.0:
            # eq_tmp = np.sum(
            #     np.logical_and(
            #         data.data[:, sens_feature] == val, gt == ylabel, preds == ylabel
            #     )
            # )
            eq_tmp = np.sum(
                [
                    1.0
                    if data.data[i, sens_feature] == val
                    and gt[i] == ylabel
                    and preds[i] == ylabel
                    else 0.0
                    for i in range(len(preds))
                ]
            )
            eo_dict[val] = eq_tmp / pos_sensitive
    return eo_dict


def neg_predict_value(
    model: BaseEstimator, data: DatasetObj, sens_feature: int, ypred: int = -1
) -> Dict[Any, float]:
    """
    Compute the negative predictive value across sensitive groups. A fair
    classifier should have a low NPV.
    """
    preds = model.predict(data.data)
    gt = data.target
    sens_feature_arr = data.data[:, sens_feature]
    sens_vals = list(set(sens_feature_arr))
    npv_dict = {}
    for val in sens_vals:
        np_tmp = None
        # neg_decisions = np.sum(
        #     np.logical_and(data.data[:, sens_feature] == val, preds == ypred)
        # )
        neg_decisions = np.sum(
            [
                1.0 if data.data[i, sens_feature] == val and preds[i] == ypred else 0.0
                for i in range(len(preds))
            ]
        )
        if neg_decisions > 0.0:
            np_tmp = np.sum(
                [
                    1.0
                    if data.data[i, sens_feature] == val
                    and preds[i] == ypred
                    and gt[i] == ypred
                    else 0.0
                    for i in range(len(preds))
                ]
            )
            npv_dict[val] = np_tmp / neg_decisions
    return npv_dict


def treatment_equality(
    model: BaseEstimator, data: DatasetObj, sens_feature: int
) -> Dict[Any, float]:
    """
    Compute Treatment Equality metric of a classifier as
    defined in Berk et al.
    """
    preds = model.predict(data.data)
    gt = data.target
    sens_feature_arr = data.data[:, sens_feature]
    sens_vals = list(set(sens_feature_arr))
    te_dict = {}
    for val in sens_vals:
        fnr_v = np.sum(
            [
                1.0
                if data.data[i, sens_feature] == val and gt[i] == 1 and preds[i] == -1
                else 0.0
                for i in range(len(preds))
            ]
        )

        fpr_v = np.sum(
            [
                1.0
                if data.data[i, sens_feature] == val and gt[i] == -1 and preds[i] == 1
                else 0.0
                for i in range(len(preds))
            ]
        )
        print(fnr_v, fpr_v)
        # fnr_v = np.sum(
        #     np.logical_and(data.data[:, sens_feature] == val, gt == 1, pred == -1)
        # )
        # fpr_v = np.sum(
        #     np.logical_and(data.data[:, sens_feature] == val, gt == -1, pred == 1)
        # )
        if fnr_v > 0.0 and fpr_v > 0.0:
            te_dict[val] = fnr_v / fpr_v
    return te_dict
