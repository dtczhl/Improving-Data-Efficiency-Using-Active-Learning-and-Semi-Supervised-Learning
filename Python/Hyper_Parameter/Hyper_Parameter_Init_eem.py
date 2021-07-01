"""
    For EEM data. LinearSVM
"""

import numpy as np

from sklearn.svm import SVC

import optuna
import sklearn
from sklearn.model_selection import KFold


class MyObjective_InitParam(object):

    def __init__(self, train_set, target, num_class, n_data, n_fold):

        self.train_set = train_set

        self.target = target

        self.n_fold = n_fold

        self.n_data = n_data

        self.best_param = None

        self.num_class = num_class

        self.accuracy = 0
        self.best_accuracy = 0

        self.loss = 0
        self.best_loss = 0

        self.best_booster = None
        self.booster = None

    def __call__(self, trial):

        self.param = trial.suggest_float('C', 1e-5, 1e2, log=True)

        train_idx = np.arange(len(self.train_set))

        train_idx_run = np.random.choice(train_idx, self.n_data)

        train_set_run = self.train_set.iloc[train_idx_run]
        train_set_run.reset_index(drop=True, inplace=True)

        target_run = self.target[train_idx_run]

        folds = KFold(n_splits=self.n_fold, shuffle=True)

        self.accuracy = 0
        for fold_, (train_idx_, val_idx_) in enumerate(folds.split(train_set_run.values, target_run)):

            print(len(train_idx_), len(val_idx_))

            train_data = train_set_run.iloc[train_idx_]
            train_label = target_run[train_idx_]

            val_data = train_set_run.iloc[val_idx_]
            val_label = target_run[val_idx_]

            model = SVC(kernel='linear', C=self.param)

            model.fit(train_data, train_label)
            pred = model.predict(val_data)
            pred_label = 1 * (pred >= 0.5)
            self.accuracy += np.sum(pred_label == val_label) / len(val_label) / self.n_fold

        return self.accuracy

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_param = self.param
            self.best_accuracy = self.accuracy