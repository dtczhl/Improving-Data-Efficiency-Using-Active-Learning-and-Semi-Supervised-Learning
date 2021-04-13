import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import optuna
from numpy import savetxt
import pickle
import os

import argparse
from argparse import RawTextHelpFormatter

import lightgbm as lgb

from copy import deepcopy

from MyObjective import MyObjective
from Query_Strategy import query_index


# ------ Configurations ------

# self_train


# take in as arguments
# query_strategy = "random"

# do not change
# n_sample_arr = list(range(3, 91))

# number of runs for each reduced number of samples
n_run = 10

# optuna number of trails
# n_trial = 5

# data directory
dire = "../data/PAW FTIR data/"

# number of folds for cross-validation. !Do not change
n_fold = 5

# do not change
start_n_sample = 10
end_n_sample = 90

path_to_param = "Hyper_Parameter/params.pkl"

# --- End of Configurations ---

# np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)

help_message = "Self Training. " \
    "Supported Methods:\n" \
    "Self Training: [random|confident|entropy]"
parser = argparse.ArgumentParser(description="Active Learning Strategies", formatter_class=RawTextHelpFormatter)
parser.add_argument("sampling_method", type=str, help=help_message)
args = parser.parse_args()

query_strategy = args.sampling_method


print("query_strategy={}\nn_run={}".format(query_strategy, n_run))

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

stand = preprocessing.StandardScaler()
data = stand.fit_transform(train_set)

le = preprocessing.LabelEncoder()
target_cf = le.fit_transform(target)

num_class = len(set(target_cf))

# Load Hyper-parameters
with open(path_to_param, "rb") as f:
    hyper_params = pickle.load(f)


def run_cv(train_set, target):

    folds = KFold(n_splits=n_fold, shuffle=True)
    preds = np.zeros([end_n_sample, len(train_set), num_class])
    pred_accuracy = np.zeros(end_n_sample)

    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        unqueried_index_set = set(train_idx)
        queried_index_set = set()

        # randomly select instances for initialization
        for _ in range(start_n_sample):
            sample_index = np.random.choice(tuple(unqueried_index_set))
            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)

        while len(queried_index_set) <= end_n_sample:

            print("\t #fold: {}/{}, #sample: {}/{}".format(fold_+1, n_fold, len(queried_index_set), end_n_sample))

            # incrementally select data instances
            queried_index_set_copy = deepcopy(queried_index_set)
            unqueried_index_set_copy = deepcopy(unqueried_index_set)
            target_copy = deepcopy(target)

            while len(unqueried_index_set_copy) > 0 and len(queried_index_set_copy) <= end_n_sample:

                train_data = lgb.Dataset(train_set.iloc[list(queried_index_set_copy)], label=target_copy[list(queried_index_set_copy)])
                valid_data = lgb.Dataset(train_set.iloc[val_idx], label=target_copy[val_idx])

                model = lgb.train(hyper_params[len(queried_index_set_copy)], train_data, valid_sets=[valid_data], verbose_eval=False)

                sample_index = query_index(model=model, train_set=train_set,
                                           queried_index_set=queried_index_set_copy,
                                           unqueried_index_set=unqueried_index_set_copy,
                                           query_strategy="selfTrain"+query_strategy, target=target_copy,
                                           val_idx=val_idx, hyper_params=hyper_params)

                target_copy[sample_index] = np.argmax(np.squeeze(model.predict(train_set.iloc[sample_index])))
                unqueried_index_set_copy.remove(sample_index)
                queried_index_set_copy.add(sample_index)

            # model evaluation
            train_data = lgb.Dataset(train_set.iloc[list(queried_index_set_copy)],
                                     label=target_copy[list(queried_index_set_copy)])
            valid_data = lgb.Dataset(train_set.iloc[val_idx], label=target_copy[val_idx])

            model = lgb.train(hyper_params[end_n_sample], train_data, valid_sets=[valid_data], verbose_eval=False)

            preds[len(queried_index_set) - 1, val_idx, :] = model.predict(train_set.iloc[val_idx])

            sample_index = np.random.choice(tuple(unqueried_index_set))
            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)

    for i_pred in range(len(preds)):
        if i_pred < start_n_sample:
            continue
        pred_label = np.argmax(preds[i_pred], axis=1)
        accuracy = np.sum(pred_label == target) / len(target)
        pred_accuracy[i_pred] = accuracy

    return pred_accuracy


result_pred = np.zeros([n_run, end_n_sample])
for i_run in range(n_run):
    print("------ Round: {}/{} -------------------------------".format(i_run+1, n_run))
    result_pred[i_run] = run_cv(train_set, target_cf)

print(result_pred)

# do not want dot in filenames
query_strategy = query_strategy.replace('.', '')
print("Saving result to ./Result/selfTrain_{}.csv".format(query_strategy))
savetxt("./Result/selfTrain_{}.csv".format(query_strategy), result_pred, delimiter=',')




