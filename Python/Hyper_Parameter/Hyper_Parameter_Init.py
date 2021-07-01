import numpy as np
import lightgbm as lgb
import optuna
import sklearn
from sklearn.model_selection import KFold


class MyObjective_InitParam(object):

    def __init__(self, train_set, target, num_class, n_data, n_fold):

        self.train_set = train_set

        self.target = target

        self.num_class = num_class

        self.n_fold = n_fold

        self.n_data = n_data

        self.best_param = None

        self.accuracy = 0
        self.best_accuracy = 0

        self.loss = 0
        self.best_loss = 0

        self.best_booster = None
        self.booster = None

    def __call__(self, trial):

        self.param = {
            'objective': 'multiclass',
            'num_class': self.num_class,
            'metric': 'multi_logloss',
            "boosting_type": "gbdt",
            "verbose": -1,
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 0.5, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 0.5, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 128, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.8, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.8, 1.0)
        }

        train_idx = np.arange(len(self.train_set))

        train_idx_set = set(train_idx)

        train_idx_run = np.random.choice(train_idx, self.n_data)

        train_set_run = self.train_set.iloc[train_idx_run]
        train_set_run.reset_index(drop=True, inplace=True)

        target_run = self.target[train_idx_run]

        folds = KFold(n_splits=self.n_fold, shuffle=True)
        pred = np.zeros([len(train_set_run), self.num_class])

        for fold_, (train_idx_, val_idx_) in enumerate(folds.split(train_set_run.values, target_run)):

            train_idx_fold_set = set(train_idx_)
            valid_idx_fold_set = train_idx_set.difference(train_idx_fold_set)
            valid_idx = list(valid_idx_fold_set)

            train_data = lgb.Dataset(train_set_run.iloc[train_idx_], label=target_run[train_idx_])

            # use all the remaining for validation
            val_data = lgb.Dataset(train_set_run.iloc[valid_idx], label=target_run[valid_idx])

            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
            gbm = lgb.train(self.param, train_data, valid_sets=[val_data], verbose_eval=False, callbacks=[pruning_callback])

            self.booster = gbm

            pred[val_idx_, :] = gbm.predict(train_set_run.iloc[val_idx_], num_iteration=gbm.best_iteration)

        gt = np.zeros((len(target_run), self.num_class), dtype=float)
        for _i in range(len(gt)):
            gt[_i][target_run[_i]] = 1

        self.loss = sklearn.metrics.log_loss(gt, pred)

        pred_label = np.argmax(pred, axis=1)
        self.accuracy = np.sum(pred_label == target_run) / len(target_run)

        return self.loss

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_booster = self.booster
            self.best_param = self.param
            self.best_loss = self.loss
            self.best_accuracy = self.accuracy