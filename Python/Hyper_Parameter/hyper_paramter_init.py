"""
    Create Initial Hyper-parameters for Plasma/LightGBM in different data sizes

    Results saved to ./Hyper_Parameter/params.pkl
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import optuna
import pickle

from Python.Hyper_Parameter.Hyper_Parameter_Init import MyObjective_InitParam

import os

# change working directory to the current file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# optuna number of trails
n_trial = 30

# data directory
dire = "../../data/PAW FTIR data/"

# number of folds for cross-validation. !Do not change
n_fold = 5

# do not change
start_n_sample = 30
end_n_sample = 90


# --- End of Configurations ---

result_hyper = {}

# np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)


print("n_trial={}".format(n_trial))

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

df["group"] = df["group"].apply(lambda x: x.split('-')[0])
df["group"] = df["group"].astype("int64")  # change to number data type

train_set = df.drop(["group"], axis=1)
target = df["group"]

stand = preprocessing.StandardScaler()
data = stand.fit_transform(train_set)
data = pd.DataFrame(data)
data.columns = train_set.columns
train_set = data

le = preprocessing.LabelEncoder()
target = le.fit_transform(target)

num_class = len(set(target))

train_idx = np.arange(len(train_set))

for i_size in range(start_n_sample, end_n_sample+1):

    print("#samples: {}/{}".format(i_size, end_n_sample))

    my_objective = MyObjective_InitParam(train_set=train_set, target=target, num_class=num_class, n_fold=n_fold,
                                         n_data=i_size)

    study = optuna.create_study(direction="minimize")
    study.optimize(my_objective, n_trials=n_trial, callbacks=[my_objective.callback])

    result_hyper[i_size] = my_objective.best_param

    print("------ ", i_size, my_objective.best_accuracy)

with open("./params.pkl", "wb") as f:
    pickle.dump(result_hyper, f)




