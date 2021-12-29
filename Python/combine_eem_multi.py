import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.semi_supervised import LabelSpreading
import optuna
from numpy import savetxt
import os

import pickle

import argparse
from argparse import RawTextHelpFormatter
from sklearn.linear_model import LogisticRegression

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
start_n_sample = 25
end_n_sample = 127
# end_n_sample - end_n_train for optuna

path_to_param = "Hyper_Parameter/params_eem_multi.pkl"

# --- End of Configurations ---

# np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)

help_message = "Combine Active Learning and Semi-supervised Learning. " \
    "Supported Methods (Active Learning):\n" \
    "Random: random;\n" \
    "Uncertainty: uncertainty_leastConfident, uncertainty_margin, uncertainty_entropy;\n" \
    "Density Weighting: density_[leastConfident|margin|entropy]_[cosine|pearson|euclidean]_[x];\n" \
    "Minimize Expected Error: minimize_[leastConfident|entropy];\n" \
    "Supported Methods (Semi_supervised Learning): labelSpread_[rbf|knn], selfTrain_random"
parser = argparse.ArgumentParser(description="Active Learning Strategies", formatter_class=RawTextHelpFormatter)
parser.add_argument("active_learning", type=str, help=help_message)
parser.add_argument("semi_supervised_learning", type=str, help=help_message)
args = parser.parse_args()

active_learning_query_strategy = args.active_learning
semi_supervised_learning_query_strategy = args.semi_supervised_learning

print("active_learning_query_strategy={}\nsemi_supervised_learning_query_strategy={}\nn_run={}"
      .format(active_learning_query_strategy, semi_supervised_learning_query_strategy, n_run))
# path to the data file
eem_df = pd.read_excel('../data/2ET_40_trim20_L.xlsx', sheet_name='Sheet1', engine='openpyxl')

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

le = preprocessing.LabelEncoder()
target_cf = le.fit_transform(target_all)

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

            if semi_supervised_learning_query_strategy.split("_")[0].lower() == 'labelspread':

                target_copy = deepcopy(target)
                target_copy[list(unqueried_index_set)] = -1

                kernel = semi_supervised_learning_query_strategy.split("_")[1]
                label_prop_model = LabelSpreading(kernel=kernel, gamma=0.1)
                label_prop_model.fit(train_set.iloc[train_idx], target_copy[train_idx])
                pred_label = label_prop_model.predict(train_set)

                pred_label[val_idx] = target[val_idx]
                pred_label[list(queried_index_set)] = target[list(queried_index_set)]
            elif semi_supervised_learning_query_strategy.split("_")[0].lower() == 'selftrain':

                # incrementally select data instances
                queried_index_set_copy = deepcopy(queried_index_set)
                unqueried_index_set_copy = deepcopy(unqueried_index_set)
                target_copy = deepcopy(target)

                while len(unqueried_index_set_copy) > 0 and len(queried_index_set_copy) <= end_n_sample:
                    train_data = train_set.iloc[list(queried_index_set_copy)]
                    train_label = target[list(queried_index_set_copy)]

                    val_data = train_set.iloc[val_idx]
                    val_label = target[val_idx]

                    model = LogisticRegression(solver='newton-cg', multi_class='multinomial',
                                               C=hyper_params[len(queried_index_set)])
                    model.fit(train_data, train_label)
                    sample_index = query_index(model=model, train_set=train_set,
                                               queried_index_set=queried_index_set_copy,
                                               unqueried_index_set=unqueried_index_set_copy,
                                               query_strategy=semi_supervised_learning_query_strategy, target=target_copy,
                                               val_idx=val_idx, hyper_params=hyper_params)
                    target_copy[sample_index] = np.argmax(np.squeeze(model.predict_proba([train_set.iloc[sample_index]])))

                    unqueried_index_set_copy.remove(sample_index)
                    queried_index_set_copy.add(sample_index)

            if semi_supervised_learning_query_strategy.split("_")[0].lower() == 'labelspread':

                train_data = train_set.iloc[train_idx]
                train_label = pred_label[train_idx]

            elif semi_supervised_learning_query_strategy.split("_")[0].lower() == 'selftrain':

                train_data = train_set.iloc[list(queried_index_set_copy)]
                train_label = target_copy[list(queried_index_set_copy)]

                val_data = train_set.iloc[val_idx]
                val_label = target[val_idx]

            model = LogisticRegression(solver='newton-cg', multi_class='multinomial',
                                       C=hyper_params[len(queried_index_set)])
            model.fit(train_data, train_label)

            preds[len(queried_index_set)-1, val_idx, :] = model.predict_proba(train_set.iloc[val_idx])

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
print("Saving result to ./Result/{}_{}_eem_multi.csv".format(active_learning_query_strategy, semi_supervised_learning_query_strategy))
savetxt("./Result/{}_{}_eem_multi.csv".format(active_learning_query_strategy, semi_supervised_learning_query_strategy), result_pred, delimiter=',')




