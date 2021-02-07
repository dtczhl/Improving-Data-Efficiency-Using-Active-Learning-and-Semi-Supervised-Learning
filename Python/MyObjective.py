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
        param = {
            'objective': 'multiclass',
            'num_class': self.num_class,
            'metric': 'multi_logloss',
            "boosting_type": "gbdt",
            "verbose": -1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 512),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100)
        }

        train_idx = self.train_idx
        train_data = lgb.Dataset(self.train_set.iloc[train_idx], label=self.target[train_idx])
        val_data = lgb.Dataset(self.train_set.iloc[self.val_idx], label=self.target[self.val_idx])

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
        gbm = lgb.train(param, train_data, valid_sets=[val_data], verbose_eval=False, callbacks=[pruning_callback])

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
            # print(study.best_trial.number, end=' ')
            self.best_booster = self.booster
