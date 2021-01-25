import os
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import lightgbm as lgb

import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt

# data directory
dire = "../data/PAW FTIR data/"

n_run = 10

n_train_sample_arr = range(10, 91, 10)
# n_train_sample_arr = [91]

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
# print(df.shape)

# "group" column refers to Y, like naocl concentration; originally it is 0ppm; I convert it to numeric number 0
df["group"] = df["group"].apply(lambda x: x.split('-')[0])
df["group"] = df["group"].astype("int64")  # change to number data type
# print(df.head())

train_set = df.drop(["group"], axis=1)
target = df["group"]
# print(train_set.shape)


# standardize the data before running PCA
stand = preprocessing.StandardScaler()
# stand.fit(train_set)
data = stand.fit_transform(train_set)

# fit the data with PCA
# pca = PCA(n_components=2)

# reduced_data is the data obtained after PCA transformation
# reduced_data = pca.fit_transform(data)
# reduced_data = pd.DataFrame(reduced_data, columns=["Dimension 1", "Dimension 2"])

le = preprocessing.LabelEncoder()
target_cf = le.fit_transform(target)
# print(data)


def run_cv(train_set, target, num_class, n_sample):
    # these are the hyperparameters for the model
    param = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.0005,
        "boosting": "gbdt",
        "feature_fraction": 1,
        "bagging_freq": 1,
        "bagging_fraction": 0.7083,
        "bagging_seed": 11,
        "lambda_l1": 0.2634,
        "random_state": 133,
        "verbose": -1
    }
    folds = KFold(n_splits=5, shuffle=True)
    oof = np.zeros([len(train_set), num_class])
    feature_importance_df = pd.DataFrame()

    # below is to split the entire dataset into train and test sets; Train on train set and evaluate on test set
    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        # print(len(val_idx))

        # val_idx = np.append(val_idx, train_idx[-1])
        # train_idx = train_idx[:-1]

        train_idx = train_idx[:n_sample]

        print(len(train_idx), len(val_idx))

        # print("fold {}".format(fold_))
        train_data = lgb.Dataset(train_set.iloc[train_idx], label=target[train_idx])
        val_data = lgb.Dataset(train_set.iloc[val_idx], label=target[val_idx])

        num_round = 10000
        clf = lgb.train(param, train_data, num_round, valid_sets=[val_data], verbose_eval=False)
        oof[val_idx, :] = clf.predict(train_set.iloc[val_idx], num_iteration=clf.best_iteration)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train_set.columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # exit(0)

    return feature_importance_df, oof


# print("Full dataset: {}".format(train_set.shape))
result_pred = np.zeros([len(n_train_sample_arr), n_run])
for i_train_sample in range(len(n_train_sample_arr)):
    n_train_sample = n_train_sample_arr[i_train_sample]

    for i_run in range(n_run):
        feature_importance_df_cf, oof_cf = run_cv(train_set, target_cf, len(set(target_cf)), n_train_sample)
        pred = np.zeros([len(oof_cf)])
        for i in range(len(oof_cf)):
            pred[i] = np.argmax(oof_cf[i])
        cm = confusion_matrix(target_cf, pred)
        total_pred_correct = sum(cm.diagonal())/np.sum(cm)

        result_pred[i_train_sample][i_run] = total_pred_correct

print(result_pred)

savetxt("result_pred.csv", result_pred, delimiter=',')





