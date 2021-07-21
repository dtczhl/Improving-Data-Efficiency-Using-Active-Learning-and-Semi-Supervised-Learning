"""
    Create Initial Hyper-parameters for each data size

    Results saved to ./Hyper_Parameter/params_eem.pkl
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import optuna
import os

import pickle

from Python.Hyper_Parameter.Hyper_Parameter_Init_eem import MyObjective_InitParam

# optuna number of trails
n_trial = 30


# number of folds for cross-validation. !Do not change
n_fold = 5

# do not change
start_n_sample = 20
end_n_sample = 57
# end_n_sample - end_n_train for optuna


# --- End of Configurations ---

result_hyper = {}

# np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)


print("n_trial={}".format(n_trial))

eem_df = pd.read_excel('../../data/se_sp_sep_260.xlsx', sheet_name='Sheet2', engine='openpyxl')

# extract sp, se, and sep
sp_df = eem_df.filter(regex='SP[0-9]+')
se_df = eem_df.filter(regex='SE[0-9]+')
sep_df = eem_df.filter(regex='SEP[0-9]+')

# transpose
sp_df = sp_df.transpose()
se_df = se_df.transpose()
sep_df = sep_df.transpose()

# add label
sp_df["Label"] = 0
se_df["Label"] = 0
sep_df["Label"] = 1

eem_new_df = pd.concat([sp_df, se_df, sep_df], ignore_index=True)

# change column names
column_names = eem_new_df.columns
new_column_names = ['Label' if x == 'Label' else 'Feature_' + str(x+1) for x in column_names]
eem_new_df.columns = new_column_names

train_set = eem_new_df.drop(["Label"], axis=1)
target = eem_new_df["Label"]

stand = preprocessing.StandardScaler()
data = stand.fit_transform(train_set)
data = pd.DataFrame(data)
data.columns = train_set.columns
train_set = data


le = preprocessing.LabelEncoder()
target = le.fit_transform(target)

# num_class = len(set(target))

train_idx = np.arange(len(train_set))

for i_size in range(start_n_sample, end_n_sample+1):

    print("#samples: {}/{}".format(i_size, end_n_sample))

    my_objective = MyObjective_InitParam(train_set=train_set, target=target, num_class=2, n_fold=n_fold,
                                         n_data=i_size)

    study = optuna.create_study(direction="maximize")
    study.optimize(my_objective, n_trials=n_trial, callbacks=[my_objective.callback])

    result_hyper[i_size] = my_objective.best_param

    print("------ ", i_size, my_objective.best_accuracy)

with open("./params_eem.pkl", "wb") as f:
    pickle.dump(result_hyper, f)




