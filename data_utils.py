from tokenize import blank_re
import numpy as np
from typing import Tuple, List, Iterable
import pandas as pd
from dataclasses import dataclass
import sklearn.preprocessing as preprocessing
from collections import namedtuple
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


@dataclass
class DatasetObj:
    data: np.ndarray
    target: np.ndarray
    s: np.ndarray = None


def load_adult(
    smaller: bool = False, scaler: bool = True
) -> Tuple[DatasetObj, DatasetObj, int]:
    """
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    """
    sens_feature = 9
    data = pd.read_csv(
        "./datasets/adult/adult.data",
        names=[
            "Age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "capital gain",
            "capital loss",
            "hours per week",
            "native-country",
            "income",
        ],
    )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        "./datasets/adult/adult.test",
        names=[
            "Age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "capital gain",
            "capital loss",
            "hours per week",
            "native-country",
            "income",
        ],
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(
        [
            "Divorced",
            "Married-AF-spouse",
            "Married-civ-spouse",
            "Married-spouse-absent",
            "Never-married",
            "Separated",
            "Widowed",
        ],
        [
            "not married",
            "married",
            "married",
            "married",
            "not married",
            "not married",
            "not married",
        ],
        inplace=True,
    )
    # categorical fields
    category_col = [
        "workclass",
        "race",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "gender",
        "native-country",
        "income",
    ]
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    target = np.array([-1.0 if val == 0 else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if smaller:
        print("A smaller version of the dataset is loaded...")
        n_sampl = len_train // 20
        data = DatasetObj(datamat[:n_sampl, :-1], target[:n_sampl])
        data_test = DatasetObj(datamat[len_train:, :-1], target[len_train:])
    else:
        print("The dataset is loaded...")
        data = DatasetObj(datamat[:len_train, :-1], target[:len_train])
        data_test = DatasetObj(datamat[len_train:, :-1], target[len_train:])
    return data, data_test, sens_feature


def load_german(
    file: str = "datasets/german_numerical-binsensitive.csv",
) -> Tuple[DatasetObj, int]:
    data = pd.read_csv(file)
    scaler = StandardScaler()
    columns = list(data.columns)
    sens_feature = columns.index("sex")
    lbl_idx = columns.index("credit")  # the label index
    data_arr = data.to_numpy()
    y = data_arr[:, lbl_idx]
    y[y == 2] = -1  # uniform classes
    X = np.delete(data_arr, lbl_idx, 1)
    Xs = scaler.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     Xs, y, random_state=0, test_size=0.2
    # )
    # data_train = DatasetObj(X_train, y_train)
    # data_test = DatasetObj(X_test, y_test)
    data = DatasetObj(Xs, y)
    columns.remove("credit")
    return data, sens_feature


def load_compas(
    file: str = "datasets/propublica-recidivism_numerical_binsensitive.csv",
) -> Tuple[DatasetObj, int]:
    """
    Load recidivism dataset used by Propublica
    """
    compas_df = pd.read_csv(file)
    features = [
        "sex",
        "age",
        "race",
        "priors_count",
        "age_cat_25 - 45",
        "age_cat_Greater than 45",
        "age_cat_Less than 25",
        "c_charge_degree_F",
        "c_charge_degree_M",
    ]
    sens_f = features.index("race")
    compas = compas_df[features]
    data = np.array(compas.values)
    labels = np.array(compas_df["two_year_recid"])
    labels[labels == 0] = -1
    scaler = StandardScaler()
    ds = scaler.fit_transform(data)
    data = DatasetObj(ds, labels)
    return data, sens_f


# def load_compas(
#     COMPAS_FILE: str = "datasets/compas-scores-two-years.csv",
# ) -> Tuple[DatasetObj, int]:
#     FEATURES_CLASSIFICATION = [
#         "age_cat",
#         "race",
#         "sex",
#         "priors_count",
#         "c_charge_degree",
#     ]
#     # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
#     CONT_VARIABLES = ["priors_count"]
#     CLASS_FEATURE = "two_year_recid"  # the decision variable
#     SENSITIVE_ATTRS = ["race"]

#     labels = []

#     df = pd.read_csv(COMPAS_FILE)
#     df = df.dropna(subset=["days_b_screening_arrest"])
#     data = df.to_dict("list")
#     for k in data.keys():
#         data[k] = np.array(data[k])

#     idx = np.logical_and(
#         data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30
#     )
#     idx = np.logical_and(idx, data["is_recid"] != -1)
#     idx = np.logical_and(idx, data["c_charge_degree"] != "O")
#     # idx = np.logical_and(idx, data["c_charge_degree"] != "O")
#     idx = np.logical_and(
#         idx,
#         np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"),
#     )

#     for k in data.keys():
#         data[k] = data[k][idx]
#     subset = df.from_dict(data)

#     y = data[CLASS_FEATURE]
#     y[y == 0] = -1  # label in {-1, 1}
#     # sens_feats = None
#     X = np.array([]).reshape(len(y), 0)
#     for attr in FEATURES_CLASSIFICATION:
#         vals = data[attr]
#         if attr in CONT_VARIABLES:
#             labels += [attr]
#             vals = [float(v) for v in vals]
#             vals = preprocessing.scale(vals)
#             vals = np.reshape(vals, (len(y), 1))
#         else:
#             unq = list(subset[attr].unique())
#             if len(unq) > 2:
#                 lbl = [f"{attr}_" + str(v) for v in unq]
#                 labels += lbl
#             else:
#                 labels += [attr]
#             lb = LabelBinarizer()
#             lb.fit(vals)
#             vals = lb.transform(vals)
#             # print(attr, unq, vals.shape)

#         # if attr in SENSITIVE_ATTRS:
#         #     sens_feats = vals.flatten()

#         X = np.hstack([X, vals])

#     assert X.shape[1] > 0, f"Invalid shape for data {X.shape}"
#     data = DatasetObj(X, y)
#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     X, y, test_size=0.2, random_state=0
#     # )
#     # data_train = DatasetObj(X_train, y_train)
#     # data_test = DatasetObj(X_test, y_test)
#     sens_feat = labels.index(SENSITIVE_ATTRS[0])
#     return data, sens_feat


def load_drug_data(
    dataset_file: str = "datasets/drug_consumption.data",
) -> Tuple[DatasetObj, int]:
    drug_df = pd.read_csv(dataset_file)
    sens_idx = 5
    x = drug_df.to_numpy()
    X_d = x[:, 1:12]
    y = x[:, 20]
    y[y == "CL0"] = -1  # drug never used
    y[y != -1] = 1  # used a drug atleast one
    y = y.astype(np.double)
    X_d[:, sens_idx][X_d[:, sens_idx] != -0.31685] = 0  # other races
    X_d[:, sens_idx][X_d[:, sens_idx] == -0.31685] = 1  # white

    scaler = StandardScaler()
    X = scaler.fit_transform(X_d)
    data = DatasetObj(X, y)
    return data, sens_idx


def load_arrhythmia_data(
    dataset_file: str = "datasets/arrhythmia.data",
) -> Tuple[DatasetObj, int]:
    arrhy_df = pd.read_csv(dataset_file)
    arrhy_df.columns = list(range(280))
    arrhy_df.drop(13, axis=1, inplace=True)
    arrhy_df[[10, 11, 12, 14]] = arrhy_df[[10, 11, 12, 14]].replace("?", np.NaN)
    arrhy_df[[10, 11, 12, 14]] = arrhy_df[[10, 11, 12, 14]].astype("float")
    arrhy_df.fillna(arrhy_df.median(), inplace=True)
    arrhy_df.columns = list(range(279))
    data = arrhy_df.values
    X = data[:, :278]
    y = data[:, 278]
    y[y != 1] = -1
    print("Label unique value: ", np.unique(y))

    sens_idx = 1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    data = DatasetObj(X, y)
    return data, sens_idx


def load_bank_data(data_f = 'datasets/bank_processed.npy', label_f = 'datasets/bank_processed_labels.npy'):
    '''
    load pre-processed bank dataset
    '''
    Xs = np.load(data_f)
    y = np.load(label_f)
    sens = 2
    data = DatasetObj(Xs, y)
    return data, sens