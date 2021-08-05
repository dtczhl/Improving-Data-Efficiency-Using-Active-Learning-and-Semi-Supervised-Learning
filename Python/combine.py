import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.semi_supervised import LabelSpreading
import optuna
from numpy import savetxt
import os

import lightgbm as lgb

import pickle

import argparse
from argparse import RawTextHelpFormatter

from Query_Strategy import query_index

from copy import deepcopy


# ------ Configurations ------

# random
# Uncertainty: uncertainty_leastConfident, uncertainty_margin, uncertainty_entropy
# Information Density: density_[leastConfident|margin|entropy]_[cosine]_[x]
# Minimizing Expected Error: minimize_leastConfident, minimize_entropy


# take in as arguments
# query_strategy = "random"

# do not change
# n_sample_arr = list(range(3, 91))

# number of runs for each reduced number of samples
n_run = 10

# data directory
dire = "../data/PAW FTIR data/"

# number of folds for cross-validation. !Do not change
n_fold = 5

# do not change
start_n_sample = 40
end_n_sample = 90
# end_n_sample - end_n_train for optuna

path_to_param = "Hyper_Parameter/params.pkl"

# --- End of Configurations ---

# np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)

help_message = "Combine Active Learning and Semi-supervised Learning. " \
    "Supported Methods (Active Learning):\n" \
    "Random: random;\n" \
    "Uncertainty: uncertainty_leastConfident, uncertainty_margin, uncertainty_entropy;\n" \
    "Density Weighting: density_[leastConfident|margin|entropy]_[cosine|pearson|euclidean]_[x];\n" \
    "Minimize Expected Error: minimize_leastConfident, minimize_entropy;\n" \
    "Supported Methods (Semi_supervised Learning): labelSpread_rbf"
parser = argparse.ArgumentParser(description="Active Learning Strategies", formatter_class=RawTextHelpFormatter)
parser.add_argument("active_learning", type=str, help=help_message)
parser.add_argument("semi_supervised_learning", type=str, help=help_message)
args = parser.parse_args()

active_learning_query_strategy = args.active_learning
semi_supervised_learning_query_strategy = args.semi_supervised_learning

print("active_learning_query_strategy={}\nsemi_supervised_learning_query_strategy={}\nn_run={}"
      .format(active_learning_query_strategy, semi_supervised_learning_query_strategy, n_run))

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

        # randomly select 10 instances for initialization
        for _ in range(start_n_sample):
            sample_index = np.random.choice(tuple(unqueried_index_set))
            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)

        while len(queried_index_set) <= end_n_sample:
            # add one sample
            print("\t #fold: {}/{}, #sample: {}/{}".format(fold_+1, n_fold, len(queried_index_set), end_n_sample))

            target_copy = deepcopy(target)
            target_copy[list(unqueried_index_set)] = -1

            kernel = semi_supervised_learning_query_strategy.split("_")[1]
            label_prop_model = LabelSpreading(kernel=kernel, gamma=0.1)
            label_prop_model.fit(train_set.iloc[train_idx], target_copy[train_idx])
            pred_label = label_prop_model.predict(train_set)

            pred_label[val_idx] = target[val_idx]
            pred_label[list(queried_index_set)] = target[list(queried_index_set)]


            # model evaluation
            train_data = lgb.Dataset(train_set.iloc[train_idx], label=pred_label[train_idx])
            valid_data = lgb.Dataset(train_set.iloc[val_idx], label=pred_label[val_idx])

            model = lgb.train(hyper_params[end_n_sample], train_data, valid_sets=[valid_data], verbose_eval=False)

            preds[len(queried_index_set)-1, val_idx, :] = model.predict(train_set.iloc[val_idx])

            sample_index = query_index(model=model, train_set=train_set,
                                       queried_index_set=queried_index_set,
                                       unqueried_index_set=unqueried_index_set,
                                       query_strategy=active_learning_query_strategy, target=target,
                                       val_idx=val_idx, hyper_params=hyper_params)

            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)

    for i_pred in range(len(preds)):
        if i_pred < start_n_sample - 1:
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
active_learning_query_strategy = active_learning_query_strategy.replace('.', '')
print("Saving result to ./Result/{}_{}.csv".format(active_learning_query_strategy, semi_supervised_learning_query_strategy))
savetxt("./Result/{}_{}.csv".format(active_learning_query_strategy, semi_supervised_learning_query_strategy), result_pred, delimiter=',')




