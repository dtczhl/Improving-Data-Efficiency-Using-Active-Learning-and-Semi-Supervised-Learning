import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import optuna
from numpy import savetxt
import pickle
import os

import matplotlib.pyplot as plt

import argparse
from argparse import RawTextHelpFormatter

from sklearn.decomposition import PCA

import random

import lightgbm as lgb

from copy import deepcopy

from MyObjective import MyObjective
from Query_Strategy import query_index

random.seed(123)
np.random.seed(123)

# data directory
dire = "../data/PAW FTIR data/"

n_run = 1

# number of folds for cross-validation. !Do not change
n_fold = 5

# do not change
start_n_sample = 40
end_n_sample = 90

path_to_param = "Hyper_Parameter/params.pkl"

# --- End of Configurations ---

# np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)

help_message = "Active Learning Sampling Method. " \
    "Supported Methods:\n" \
    "Random: random;\n" \
    "Uncertainty: uncertainty_leastConfident, uncertainty_margin, uncertainty_entropy;\n" \
    "Density Weighting: density_[leastConfident|margin|entropy]_[cosine|pearson|euclidean]_[x];\n" \
    "Minimize Expected Error: minimize_leastConfident, minimize_entropy"
parser = argparse.ArgumentParser(description="Active Learning Strategies", formatter_class=RawTextHelpFormatter)
parser.add_argument("sampling_method", type=str, help=help_message)
args = parser.parse_args()

query_strategy = args.sampling_method

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

        queried_index_list = []

        # randomly select 10 instances for initialization
        for _ in range(start_n_sample):
            sample_index = np.random.choice(tuple(unqueried_index_set))
            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)
            queried_index_list.append(sample_index)

        while len(queried_index_set) <= end_n_sample:
            # add one sample
            print("\t #fold: {}/{}, #sample: {}/{}".format(fold_+1, n_fold, len(queried_index_set), end_n_sample))

            train_data = lgb.Dataset(train_set.iloc[queried_index_list], label=target[queried_index_list])
            valid_data = lgb.Dataset(train_set.iloc[val_idx], label=target[val_idx])

            model = lgb.train(hyper_params[len(queried_index_list)], train_data, valid_sets=[valid_data], verbose_eval=False)

            preds[len(queried_index_list)-1, val_idx, :] = model.predict(train_set.iloc[val_idx])

            sample_index = query_index(model=model, train_set=train_set,
                                       queried_index_set=queried_index_list,
                                       unqueried_index_set=unqueried_index_set,
                                       query_strategy=query_strategy, target=target,
                                       val_idx=val_idx, hyper_params=hyper_params)

            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)

            queried_index_list.append(sample_index)

        # print(sample_seq)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(train_set.iloc[queried_index_list])
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])

        principalDf["target"] = target[queried_index_list]
        finalDf = principalDf

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=24)
        ax.set_ylabel('Principal Component 2', fontsize=24)

        # TITLE
        ax.set_title('Uncertainty Sampling', fontsize=24, fontweight="bold")

        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)

        targets = [0, 1, 2, 3]
        colors = ['r', 'g', 'b', 'y']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c=color
                       , s=100)
        ax.legend(["Dosage 1", "Dosage 2", "Dosage 3", "Dosage 4"], fontsize=20)
        ax.grid()

        for i in range(0, start_n_sample):
            plt.text(finalDf.loc[i, 'principal component 1']-1.6,
                     finalDf.loc[i, 'principal component 2']-1.5, s='*', fontsize=20)

        for i in range(start_n_sample, start_n_sample + 10):
            plt.text(finalDf.loc[i, 'principal component 1']-2.2,
                     finalDf.loc[i, 'principal component 2']-3.5, s=str(i-start_n_sample+1), fontsize=20)


        # plt.show()
        plt.savefig("../Matlab/Image/sample_seq_{}.png".format(query_strategy))

        exit()

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