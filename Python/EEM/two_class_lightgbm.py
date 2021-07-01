"""
    sp vs sep, se vs sep, and sp + se vs sep
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold

import lightgbm as lgb

import os

# number of runs
n_run = 10

# number of folds for cross-validation.
n_fold = 5

# path to the data file
eem_df = pd.read_excel('../../data/se_sp_sep_260.xlsx', sheet_name='Sheet2', engine='openpyxl')

# save results
# save_file = './EEM/eem_binary_result.csv'
# f = open(save_file, "w")
# f.write("Model,sp-sep,se-sep\n")

# extract sp, se, and sep
sp_df = eem_df.filter(regex='SP[0-9]+')
se_df = eem_df.filter(regex='SE[0-9]+')
sep_df = eem_df.filter(regex='SEP[0-9]+')

# transpose
sp_df = sp_df.transpose()
se_df = se_df.transpose()
sep_df = sep_df.transpose()

# sp + se vs sep
sp_df["Label"] = 0
se_df["Label"] = 0
sep_df["Label"] = 1
sp_se_sep_df = pd.concat([sp_df, se_df, sep_df], ignore_index=True)


# change column names
column_names = sp_se_sep_df.columns
new_column_names = ['Label' if x == 'Label' else 'Feature_' + str(x+1) for x in column_names]
sp_se_sep_df.columns = new_column_names

# training data for sp+se vs sep
train_set_sp_se_sep = sp_se_sep_df.drop(["Label"], axis=1)
target_sp_se_sep = sp_se_sep_df["Label"]

# data preprocessing
#  normalization
stand = preprocessing.StandardScaler()
data_sp_se_sep = stand.fit_transform(train_set_sp_se_sep)
data_sp_se_sep = pd.DataFrame(data_sp_se_sep)
data_sp_se_sep.columns = train_set_sp_se_sep.columns

# re-labeling
le = preprocessing.LabelEncoder()
target_cf_sp_se_sep = le.fit_transform(target_sp_se_sep)


def run_cv(train_set, target):

    param = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        "verbose": -1,
        "bagging_freq": 1
    }


    folds = KFold(n_splits=n_fold, shuffle=True)

    accuracy = 0
    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        train_data = lgb.Dataset(train_set.iloc[train_idx], label=target[train_idx])
        valid_data = lgb.Dataset(train_set.iloc[val_idx], label=target[val_idx])

        model = lgb.train(params=param, train_set=train_data, valid_sets=[valid_data], verbose_eval=False)
        pred = model.predict(train_set.iloc[val_idx])
        pred_label = 1 * (pred >= 0.5)
        accuracy += np.sum(pred_label == target[val_idx]) / len(target[val_idx]) / n_fold

    return accuracy


result_sp_se = 0
for i_run in range(n_run):
    print(i_run)
    result_sp_se += run_cv(data_sp_se_sep, target_cf_sp_se_sep) / n_run

print(result_sp_se)


