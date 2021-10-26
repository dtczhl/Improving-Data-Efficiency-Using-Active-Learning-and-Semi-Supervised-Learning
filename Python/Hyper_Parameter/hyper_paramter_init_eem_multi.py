"""
    Create Initial Hyper-parameters for Plasma/LightGBM in different data sizes

    Results saved to ./Hyper_Parameter/params.pkl
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import optuna
import pickle

from Python.Hyper_Parameter.Hyper_Parameter_Init_eem_multi import MyObjective_InitParamEEMMulti

import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# change working directory to the current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# optuna number of trails
n_trial = 30

# number of folds for cross-validation. !Do not change
n_fold = 5

# do not change
start_n_sample = 15
end_n_sample = 128


# --- End of Configurations ---

result_hyper = {}

# np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)


print("n_trial={}".format(n_trial))

# path to the data file
eem_df = pd.read_excel('../../data/2ET_40_trim20_L.xlsx', sheet_name='Sheet1', engine='openpyxl')

# extract sp, se, and sep
ep_df = eem_df.filter(regex='2EP[0-9]+')
p_df = eem_df.filter(regex='1P[0-9]+')
e_df = eem_df.filter(regex='2E[0-9]+')
l_df = eem_df.filter(regex='2LP[0-9]+')


# transpose
ep_df = ep_df.transpose()
p_df = p_df.transpose()
e_df = e_df.transpose()
l_df = l_df.transpose()

# add label
ep_df["Label"] = 0
p_df["Label"] = 1
e_df["Label"] = 2
l_df["Label"] = 3

# complete datasets for sp-sep and se-sep
all_df = pd.concat([ep_df, p_df, e_df, l_df], ignore_index=True)

# change column names
column_names = all_df.columns
new_column_names = ['Label' if x == 'Label' else 'Feature_' + str(x+1) for x in column_names]
all_df.columns = new_column_names

# training data for sp vs sep
train_set_all = all_df.drop(["Label"], axis=1)
target_all = all_df["Label"]

# data preprocessing
#  normalization
stand = preprocessing.StandardScaler()
data_all = stand.fit_transform(train_set_all)
data = pd.DataFrame(data_all)
data.columns = train_set_all.columns
train_set = data

# re-labeling
le = preprocessing.LabelEncoder()
target = le.fit_transform(target_all)

num_class = len(set(target))

train_idx = np.arange(len(train_set))

for i_size in range(start_n_sample, end_n_sample+1):

    print("#samples: {}/{}".format(i_size, end_n_sample))

    my_objective = MyObjective_InitParamEEMMulti(train_set=train_set, target=target, num_class=num_class, n_fold=n_fold,
                                                 n_data=i_size)

    study = optuna.create_study(direction="maximize")
    study.optimize(my_objective, n_trials=n_trial, callbacks=[my_objective.callback])

    result_hyper[i_size] = my_objective.best_param

    print("------ ", i_size, my_objective.best_accuracy)

with open("./params_eem_multi.pkl", "wb") as f:
    pickle.dump(result_hyper, f)




