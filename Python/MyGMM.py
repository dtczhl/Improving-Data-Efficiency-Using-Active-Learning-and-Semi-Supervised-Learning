import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture

from scipy.stats import multivariate_normal

class MyGMM(object):

    def __init__(self, n_component, covariance, weight, tol=1e-3, max_iter=100):

        self.n_component = n_component
        self.tol = tol
        self.max_iter = max_iter
        self.covariance = covariance
        self.weight = weight

        self.Pi = None
        self.Mu = None
        self.Sigma = None

        self.n_sample = None
        self.n_feature = None

    def mean_variance_pi(self, X, y, y_pred=None):

        # if y_pred is None, it is for initialization

        # initialize
        if y_pred is None:
            for i_component in range(self.n_component):
                X_component = X[y == i_component]
                self.Pi[i_component] = len(X_component) / np.sum(y != -1)
                if len(X_component) < 1:
                    X_component = 2 * (np.random.random(size=(1, self.n_feature)) - 0.5)
                self.Mu[i_component] = np.mean(X_component, axis=0)
                self.Sigma[i_component] = np.cov(X_component, rowvar=False)
        else:
            for i_component in range(self.n_component):
                X_component = X[y_pred == i_component]
                y_component = y[y_pred == i_component]
                self.Pi[i_component] = len(X_component) / self.n_sample
                n_labeled = np.sum(y_component != -1)
                n_unlabeled = np.sum(y_component == -1)
                # print(n_labeled)
                # print(n_unlabeled)
                # print(self.weight)
                w = 1/(n_labeled + self.weight * n_unlabeled)
                v1 = w * n_labeled * np.mean(X_component[y_component != -1], axis=0)
                v2 = self.weight * w * n_unlabeled * np.mean(X_component[y_component == -1], axis=0)
                if not np.isnan(v1).any() and not np.isnan(v2).any():
                    self.Mu[i_component] = v1 + v2
                elif not np.isnan(v1).any():
                    self.Mu[i_component] = v1
                elif not np.isnan(v2).any():
                    self.Mu[i_component] = v2
                else:
                    # all nans
                    pass
                # self.Mu[i_component] = w * n_labeled * np.mean(X_component[y_component != -1], axis=0) \
                #                        + self.weight * w * n_unlabeled * np.mean(X_component[y_component == -1], axis=0)
                # print(self.Mu[i_component])

                v1 = w * n_labeled * np.cov(X_component[y_component != -1], rowvar=False, bias=True)
                v2 = self.weight * w * n_unlabeled * np.cov(X_component[y_component == -1], rowvar=False, bias=True)
                if not np.isnan(v1).any() and not np.isnan(v2).any():
                    self.Sigma[i_component] = v1 + v2
                    if 1 - n_labeled * w ** 2 - n_unlabeled * w ** 2 * self.weight ** 2 != 0:
                        self.Sigma[i_component] /= (1 - n_labeled * w ** 2 - n_unlabeled * w ** 2 * self.weight ** 2)
                elif not np.isnan(v1).any():
                    self.Sigma[i_component] = v1
                    if 1 - n_labeled * w ** 2 - n_unlabeled * w ** 2 * self.weight ** 2 != 0:
                        self.Sigma[i_component] /= (1 - n_labeled * w ** 2 - n_unlabeled * w ** 2 * self.weight ** 2)
                elif not np.isnan(v2).any():
                    self.Sigma[i_component] = v2
                    if 1 - n_labeled * w ** 2 - n_unlabeled * w ** 2 * self.weight ** 2 != 0:
                        self.Sigma[i_component] /= (1 - n_labeled * w ** 2 - n_unlabeled * w ** 2 * self.weight ** 2)
                else:
                    # all nans
                    pass

                # self.Sigma[i_component] = w * n_labeled * np.cov(X_component[y_component != -1], rowvar=False, bias=True) \
                #                           + self.weight * w * n_unlabeled * np.cov(X_component[y_component == -1], rowvar=False, bias=True)
                # self.Sigma[i_component] /= (1 - n_labeled * w**2 - n_unlabeled * w**2 * self.weight**2)

                # print(self.Sigma[i_component])


    def fit(self, X, y):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        self.n_sample = X.shape[0]
        self.n_feature = X.shape[1]

        self.Pi = np.zeros(shape=self.n_component)
        self.Mu = np.zeros(shape=(self.n_component, self.n_feature))
        self.Sigma = np.zeros(shape=(self.n_component, self.n_feature, self.n_feature))

        # initialize
        self.mean_variance_pi(X, y, y_pred=None)

        t = 0
        while t < self.max_iter:
            t += 1

            # E-step
            # print("t = {} E".format(t))
            y_pred = np.zeros(shape=y.shape)

            Gaussian_model = {}
            for i_component in range(self.n_component):
                Gaussian_model[i_component] = multivariate_normal(
                    mean=self.Mu[i_component], cov=self.Sigma[i_component], allow_singular=True)

            for i_sample in range(self.n_sample):
                if y[i_sample] != -1:
                    y_pred[i_sample] = y[i_sample]
                else:
                    _pred_label = -1
                    _pred_prob = -1
                    for i_component in range(self.n_component):
                        _value = Gaussian_model[i_component].pdf(X[i_sample])
                        if _value > _pred_prob:
                            _pred_prob = _value
                            _pred_label = i_component
                    y_pred[i_sample] = _pred_label

            # M-step
            # print("t = {} M".format(t))
            self.mean_variance_pi(X, y, y_pred=y_pred)

        #
        #
        #
        #
        # R = np.zeros(shape=(n_sample, n_component))
        # # Gaussian_model = dict()
        # Gaussian_value = np.zeros(shape=(n_component))
        #
        # # initialize mu, sigma, pi
        # for i_component in range(n_component):
        #     X_component = X[y == i_component]
        #     if len(X_component) < 1:
        #         X_component = 2 * (np.random.random(size=(1, n_feature)) - 0.5)
        #     Mu[i_component] = np.mean(X_component, axis=0)
        #     Sigma[i_component] = np.cov(X_component, rowvar=False)
        #     # Pi[i_component] = len(X_component) / len(X_labeled)
        #     Pi[i_component] = len(X_component) / np.sum(y != -1)
        #
        # # Initialize R for labeled samples
        # for i_sample in range(n_sample):
        #     if y[i_sample] != -1:
        #         R[i_sample][y[i_sample]] = 1
        #
        # error = np.inf
        # # error_old = -np.inf
        # p_D = np.nan
        #
        # t = 0
        #
        # while error > self.tol and t < self.max_iter:
        #
        #     t += 1
        #
        #     # ------ E step ------
        #     for i_component in range(n_component):
        #         self.Gaussian_model[i_component] = multivariate_normal(
        #             mean=Mu[i_component], cov=Sigma[i_component], allow_singular=True)
        #
        #     for i_sample in range(n_sample):
        #         if y[i_sample] != -1:
        #             continue
        #
        #         for j_component in range(n_component):
        #             Gaussian_value[j_component] = self.Gaussian_model[j_component].pdf(X[i_sample])
        #
        #         P_x = np.dot(Gaussian_value, Pi)
        #
        #         for j_component in range(n_component):
        #             R[i_sample][j_component] = Pi[j_component] * Gaussian_value[j_component] / P_x
        #
        #     # ------ M step ------
        #
        #     # L
        #     L = np.sum(R, axis=0)
        #
        #     # Mu
        #     for i_component in range(n_component):
        #         Mu_component = np.zeros(shape=[1, n_feature])
        #         for j_sample in range(n_sample):
        #             Mu_component += R[j_sample][i_component] * X[j_sample]
        #         Mu[i_component] = Mu_component / L[i_component]
        #
        #     # Sigma
        #     for i_component in range(n_component):
        #         Sigma_component = np.zeros(shape=[n_feature, n_feature])
        #         for j_sample in range(n_sample):
        #             _temp = X[j_sample] - Mu[i_component]
        #             _temp = np.expand_dims(_temp, axis=0)
        #             _temp = R[j_sample][i_component] * np.matmul(_temp.T, _temp)
        #             Sigma_component += _temp
        #         Sigma[i_component] = Sigma_component / L[i_component]
        #
        #     # Pi
        #     Pi = L / n_sample
        #
        #     # Loss
        #     _p_D = 0
        #     for i_sample in range(n_sample):
        #
        #         if y[i_sample] != -1:
        #             value = self.Gaussian_model[y[i_sample]].pdf(X[i_sample]) * Pi[y[i_sample]]
        #             _p_D += np.log(value)
        #
        #         else:
        #             _temp = 0
        #             for j_component in range(n_component):
        #                 # print("---", self.Gaussian_model[j_component].pdf(X[i_sample]) * Pi[j_component])
        #                 value = self.Gaussian_model[j_component].pdf(X[i_sample]) * Pi[j_component]
        #                 _temp += value
        #             _p_D += np.log(_temp)
        #
        #     if not np.isnan(p_D):
        #         error = np.abs(_p_D - p_D)
        #
        #     p_D = _p_D
        #     print(p_D, error)


    def predict(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        n_sample = len(X)
        n_component = self.n_component

        pred = np.zeros(shape=n_sample)

        Gaussian_model = {}
        for i_component in range(self.n_component):
            Gaussian_model[i_component] = multivariate_normal(
                mean=self.Mu[i_component], cov=self.Sigma[i_component], allow_singular=True)

        for i_sample in range(n_sample):
            prob = -np.inf
            pred_class = -1
            for j_component in range(n_component):
                if Gaussian_model[j_component].pdf(X[i_sample]) > prob:
                    prob = Gaussian_model[j_component].pdf(X[i_sample])
                    pred_class = j_component
            pred[i_sample] = pred_class

        return pred
