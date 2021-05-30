from sklearn.base import BaseEstimator
import numpy as np
import time
import cvxopt
import cvxopt.solvers as solvers
from cvxopt import matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# from data_utils import load_dataset


def linear_kernel(x, y):
    return np.dot(x, y.T)


class FairSVM(BaseEstimator):
    """
    Implements a fairer SVM based on a
    2-dimensional constraint over the
    cluster of sensitive features.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma=0.01, sens_feat=None, ftol=1e-2):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.sens_feat = sens_feat
        self.ftol = ftol
        self.w = None
        # self.w_fair = None

    def _fit_fair_svm(self, X, y, kern_func):
        if self.sens_feat is None:
            raise ValueError("Sensitive Feature cannot be NULL for FairSVM")
        N_samp, N_feats = X.shape[0], X.shape[1]
        sens_feature = X[:, self.sens_feat]
        sens_vals = list(set(sens_feature))
        idx_a = [
            i for i in range(len(y)) if y[i] == 1 and sens_feature[i] == sens_vals[0]
        ]
        idx_b = [
            i for i in range(len(y)) if y[i] == 1 and sens_feature[i] == sens_vals[1]
        ]
        Na, Nb = len(idx_a), len(idx_b)
        K = kern_func(X, X)
        self.K = K
        D = np.outer(y, y) * K  # yiyjK(xi, xj)
        P = matrix(D)
        q = matrix(np.ones(N_samp) * -1)

        # box constraint 0 <= \alpha <= C
        tmp1 = np.diag(np.ones(N_samp) * -1)
        tmp2 = np.identity(N_samp)
        G = np.vstack((tmp1, tmp2))
        tmp1 = np.zeros(N_samp)
        tmp2 = np.ones(N_samp) * self.C
        h = np.hstack((tmp1, tmp2))
        G = matrix(G)
        h = matrix(h)

        # fairness constraint added to affine Ax = b
        mu_a = np.sum(K[idx_a], axis=0) / Na
        mu_b = np.sum(K[idx_b], axis=0) / Nb
        var_a = np.sum((K[idx_a] - mu_b.T) ** 2, axis=0) / (Na - 1)
        var_b = np.sum((K[idx_b] - mu_a.T) ** 2, axis=0) / (Nb - 1)
        Va = 1 / np.sqrt(var_a) * mu_a
        Vb = 1 / np.sqrt(var_b) * mu_b
        # Va = np.hstack([mu_a, var_a]).reshape(mu_a.shape[0], 2)
        # Vb = np.hstack([mu_b, var_b]).reshape(mu_b.shape[0], 2)
        V = Va - Vb
        self.V = V
        fairline = matrix(y * V, (1, N_samp), "d")

        A = matrix(y.astype(np.double), (1, N_samp), "d")
        A = matrix(np.vstack([A, fairline]))
        b = matrix([0.0, self.ftol])
        solvers.options["show_progress"] = False
        solution = solvers.qp(P, q, G, h, A, b)
        return solution["x"]

    def fit(self, X_train, y_train):
        # todo: linear kenrel
        if self.kernel == "rbf":
            self.fkern = lambda x, y: rbf_kernel(x, y, gamma=self.gamma)
        elif self.kernel == "linear":
            self.fkern = linear_kernel
        else:
            raise ValueError("More support coming soon!")
        alphas = self._fit_fair_svm(X_train, y_train, self.fkern)
        alphas = np.array(alphas).flatten()
        svecs = alphas > 1e-7  # sv mask
        inds = np.arange(len(alphas))[svecs]
        X_sv = X_train[svecs]
        y_sv = y_train[svecs]
        alphas = alphas[svecs]
        self.bias = 0
        for ii in range(len(alphas)):
            self.bias += y_sv[ii]
            self.bias -= np.sum(alphas * y_sv * self.K[inds[ii], svecs])
        self.bias = self.bias / len(alphas)

        self.alphas = alphas
        self.svs = svecs
        self.sv_x = X_train[svecs]
        self.sv_y = y_train[svecs]
        if self.kernel == "linear":
            self.w = np.zeros(X_train.shape[1])
            for i in range(len(self.alphas)):
                self.w += alphas[i] * y_sv[i] * X_sv[i]
        # print("Support vector X: ", self.sv_x.shape)
        # print("Support vector y: ", self.sv_y.shape)
        # print("Found alphas: ", self.alphas.shape)

    def _project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.bias
        else:
            XSV = self.fkern(X, self.sv_x)
            # print(XSV.shape)
            y_pred = [
                np.sum(np.multiply(np.multiply(self.alphas, self.sv_y), XSV[i, :]))
                for i in range(len(X))
            ]
            return y_pred + self.bias

    def _decision(self, X):
        return self._project(X)

    def predict(self, X_test):
        return np.sign(self._decision(X_test))

    def score(self, X_test, y_test):
        predict = self.predict(X_test)
        acc = accuracy_score(y_test, predict)
        return acc


def unit_test_1():
    data_train, data_test, sens_feat = load_dataset("adult", smaller=True)
    N_train = data_train.data.shape[0]
    print("Train set shape: ", data_train.data.shape)
    print("Test set shape: ", data_test.data.shape)
    start = time.time()
    fsvm = FairSVM(sens_feat=sens_feat)
    fsvm.fit(data_train.data, data_train.target)
    train_time = time.time() - start
    print(f"Trained {N_train} pts in: {train_time} s")
    start = time.time()
    y_pred = fsvm.predict(data_test.data[:N_train])
    print(f"Inference in {time.time() - start} s")
    # print(np.unique(y_pred))
    # print(np.unique(data_test.target[:N_train]))
    print("Accuracy: ", accuracy_score(data_test.target[:N_train], y_pred))


def unit_test_2():
    param_grid = [
        {"C": [0.1, 1.0, 10.0], "gamma": [0.001, 0.01, 0.1, 1.0], "kernel": ["rbf"]}
    ]
    data_train, data_test, sens_feat = load_dataset("adult", smaller=True)
    N_train = data_train.data.shape[0]
    print("Train set shape: ", data_train.data.shape)
    print("Test set shape: ", data_test.data.shape)
    fsvm = FairSVM(sens_feat=sens_feat)
    clf = GridSearchCV(fsvm, param_grid=param_grid, n_jobs=8)
    clf.fit(data_train.data, data_train.target)
    print(f"Best estimator: {clf.best_estimator_}")


def main():
    # unit testing
    print("Running unit test 1")
    unit_test_1()
    print("Running unit test 2")
    unit_test_2()


if __name__ == "__main__":
    main()
