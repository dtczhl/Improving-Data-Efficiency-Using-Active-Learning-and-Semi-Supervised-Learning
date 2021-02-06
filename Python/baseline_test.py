import os
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import optuna

import sklearn.metrics

import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt


optuna.logging.set_verbosity(optuna.logging.FATAL)

# data directory
dire = "../data/PAW FTIR data/"

n_run = 1

# n_train_sample_arr = range(10, 91, 10)
n_train_sample_arr = [90]

filesnames = os.listdir(dire)
filesnames.sort(key=lambda x: int(x.split('-')[0]))

# Below is to read all files from the directory into Pandas DataFrame
df = pd.DataFrame()
for i_file in range(len(filesnames)):
    with open(os.path.join(dire, filesnames[i_file])) as f:
        for i_row, row in enumerate(f):
            if i_row < 4:
                continue
            else:
                row = row.replace('\t', ' ')
                row = row.replace('\n', ' ')
                temp = row.split()
                col_name = float(temp[0])
                col_value = float(temp[1])
                df.loc[i_file, col_name] = col_value
        df.loc[i_file, 'group'] = filesnames[i_file][0:3]

# "group" column refers to Y, like naocl concentration; originally it is 0ppm; I convert it to numeric number 0
df["group"] = df["group"].apply(lambda x: x.split('-')[0])
df["group"] = df["group"].astype("int64")  # change to number data type

train_set = df.drop(["group"], axis=1)
target = df["group"]

stand = preprocessing.StandardScaler()
data = stand.fit_transform(train_set)
le = preprocessing.LabelEncoder()
target_cf = le.fit_transform(target)
# print(data)


def run_cv(train_set, target, num_class, n_sample):

    class Objective(object):
        def __init__(self, train_set, target, num_class, n_sample, train_idx, val_idx):
            self.train_set = train_set
            self.target = target
            self.num_class = num_class
            self.n_sample = n_sample
            self.train_idx = train_idx
            self.val_idx = val_idx

        def __call__(self, trial):
            param = {
                'objective': 'multiclass',
                'num_class': self.num_class,
                'metric': 'multi_logloss',
                "boosting_type": "gbdt",
                "verbose": -1
            }

            train_idx = self.train_idx[:self.n_sample]
            train_data = lgb.Dataset(self.train_set.iloc[train_idx], label=self.target[train_idx])
            val_data = lgb.Dataset(self.train_set.iloc[self.val_idx], label=target[self.val_idx])

            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
            gbm = lgb.train(param, train_data, valid_sets=[val_data], verbose_eval=False, callbacks=[pruning_callback])

            preds = gbm.predict(self.train_set.iloc[self.val_idx])
            gt_target = self.target[self.val_idx]
            gt = np.zeros((len(gt_target), num_class), dtype=float)
            for _i in range(len(gt)):
                gt[_i][gt_target[_i]] = 1

#############################
            print(preds)
            print(gt)
            exit()

            loss = sklearn.metrics.log_loss(self.target[self.val_idx], preds)
            return loss

    folds = KFold(n_splits=5, shuffle=True)
    oof = np.zeros([len(train_set), num_class])
    feature_importance_df = pd.DataFrame()

    # below is to split the entire dataset into train and test sets; Train on train set and evaluate on test set
    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), direction="minimize")
        study.optimize(Objective(train_set=train_set, target=target, num_class=num_class,
                       n_sample=n_sample, train_idx=train_idx, val_idx=val_idx), n_trials=100)

        print(study.best_params)

        exit()

        oof[val_idx, :] = model.predict(train_set.iloc[val_idx], num_iteration=model.best_iteration)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train_set.columns
        # fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["importance"] = model.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    return feature_importance_df, oof


# print("Full dataset: {}".format(train_set.shape))
result_pred = np.zeros([len(n_train_sample_arr), n_run])
for i_train_sample in range(len(n_train_sample_arr)):
    n_train_sample = n_train_sample_arr[i_train_sample]

    for i_run in range(n_run):

        print("\n----- #samples: {}, #run: {}/{}".format(n_train_sample, i_run+1, n_run))

        feature_importance_df_cf, oof_cf = run_cv(train_set, target_cf, len(set(target_cf)), n_train_sample)
        pred = np.zeros([len(oof_cf)])
        for i in range(len(oof_cf)):
            pred[i] = np.argmax(oof_cf[i])
        cm = confusion_matrix(target_cf, pred)
        total_pred_correct = sum(cm.diagonal())/np.sum(cm)

        result_pred[i_train_sample][i_run] = total_pred_correct

print(result_pred)

savetxt("baseline_result_pred.csv", result_pred, delimiter=',')





