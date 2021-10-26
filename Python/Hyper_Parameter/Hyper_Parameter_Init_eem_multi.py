import numpy as np
import optuna
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


class MyObjective_InitParamEEMMulti(object):

    def __init__(self, train_set, target, num_class, n_data, n_fold):

        self.train_set = train_set

        self.target = target

        self.num_class = num_class

        self.n_fold = n_fold

        self.n_data = n_data

        self.best_param = None

        self.accuracy = 0
        self.best_accuracy = 0

    def __call__(self, trial):

        self.param = trial.suggest_float('C', 1e-5, 1e2, log=True)

        train_idx = np.arange(len(self.train_set))
        train_idx = np.random.permutation(train_idx)

        train_set_run = self.train_set.iloc[train_idx]
        train_set_run.reset_index(drop=True, inplace=True)

        target_run = self.target[train_idx]

        folds = KFold(n_splits=self.n_fold, shuffle=True)
        self.accuracy = 0

        for fold_, (train_idx_, val_idx_) in enumerate(folds.split(train_set_run.values, target_run)):

            train_idx_ = train_idx_[:self.n_data]

            train_data = train_set_run.iloc[train_idx_]
            train_label = target_run[train_idx_]

            val_data = train_set_run.iloc[val_idx_]
            val_label = target_run[val_idx_]

            model = LogisticRegression(solver='newton-cg', multi_class='multinomial', C=self.param)
            model.fit(train_data, train_label)
            pred_label = model.predict(val_data)

            self.accuracy += np.sum(pred_label == val_label) / len(val_label) / self.n_fold

        return self.accuracy

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_param = self.param
            self.best_accuracy = self.accuracy
