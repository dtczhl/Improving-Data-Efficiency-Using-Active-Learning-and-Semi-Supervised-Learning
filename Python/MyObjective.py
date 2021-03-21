import numpy as np
import lightgbm as lgb
import optuna
import sklearn


class MyObjective(object):

    def __init__(self, train_set, target, num_class, train_idx, val_idx):
        self.train_set = train_set
        self.target = target
        self.num_class = num_class
        self.train_idx = train_idx
        self.val_idx = val_idx

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
            # # "lambda_l1": 0.2634,
            # # "max_depth": 10,
            # # "min_data_in_leaf": 2,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 0.5, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 0.5, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 128, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.8, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.8, 1.0),
            # "bagging_freq": trial.suggest_int("bagging_freq", 1, 2),
            # "bagging_freq": 1,
            # "min_child_samples": trial.suggest_int("min_child_samples", 10, 20)
        }

        train_idx = self.train_idx
        train_data = lgb.Dataset(self.train_set.iloc[train_idx], label=self.target[train_idx])
        val_data = lgb.Dataset(self.train_set.iloc[self.val_idx], label=self.target[self.val_idx])

        self.pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
        gbm = lgb.train(self.param, train_data, valid_sets=[val_data], verbose_eval=False, callbacks=[self.pruning_callback])

        self.booster = gbm

        preds = gbm.predict(self.train_set.iloc[self.val_idx])
        gt_target = self.target[self.val_idx]
        gt = np.zeros((len(gt_target), self.num_class), dtype=float)
        for _i in range(len(gt)):
            gt[_i][gt_target[_i]] = 1
        # loss = sklearn.metrics.log_loss(self.target[self.val_idx], preds)
        loss = sklearn.metrics.log_loss(gt, preds)

        return loss

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_booster = self.booster

