import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import optuna
from numpy import savetxt
import os

import argparse
from argparse import RawTextHelpFormatter

from MyObjective import MyObjective
from Query_Strategy import query_index


# ------ Configurations ------

# random
# Uncertainty: uncertainty_leastConfident, uncertainty_margin, uncertainty_entropy
# Information Density: density_[leastConfident | entropy]_[cosine]_[x]

# take in as arguments
# query_strategy = "random"

# do not change
n_sample_arr = list(range(3, 91))

# number of runs for each reduced number of samples
# 30
n_run = 1

# optuna number of trails
n_trial = 20

# data directory
dire = "../data/PAW FTIR data/"

# number of folds for cross-validation. !Do not change
n_fold = 5

# do not change
start_n_sample = 10
end_n_sample = 90

# --- End of Configurations ---

np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)

help_message = "Active Learning Sampling Method. " \
    "Supported Methods:\n" \
    "batch_random:\n" \
    "random;\n" \
    "Uncertainty: uncertainty_leastConfident, uncertainty_margin, uncertainty_entropy;\n" \
    "Density Weighting: density_[leastConfident|margin|entropy]_[cosine|pearson]_[x];"
parser = argparse.ArgumentParser(description="Active Learning Strategies", formatter_class=RawTextHelpFormatter)
parser.add_argument("sampling_method", type=str, help=help_message)
args = parser.parse_args()

query_strategy = args.sampling_method

print("query_strategy={}\nn_run={}\nn_trial={}\nstart_n_sample={}"
      .format(query_strategy, n_run, n_trial, start_n_sample))

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


def run_cv(train_set, target, num_class, n_sample_arr):

    folds = KFold(n_splits=n_fold, shuffle=True)
    oof = np.zeros([end_n_sample, len(train_set), num_class])

    result_pred = np.zeros(end_n_sample)

    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        unqueried_index_set = set(train_idx)
        queried_index_set = set()

        # randomly select instances for initialization
        for _ in range(start_n_sample):
            sample_index = np.random.choice(tuple(unqueried_index_set))
            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)

        my_objective = MyObjective(train_set=train_set, target=target, num_class=num_class,
                                   train_idx=list(queried_index_set), val_idx=val_idx)

        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.RandomSampler(), direction="minimize")
        study.optimize(my_objective, n_trials=n_trial, callbacks=[my_objective.callback])
        model = my_objective.best_booster
        oof[len(queried_index_set), val_idx, :] = model.predict(train_set.iloc[val_idx], num_iteration=model.best_iteration)

        while len(queried_index_set) < end_n_sample:

            print("\t #sample: {}/{}".format(len(queried_index_set)+1, end_n_sample))

            # for batch random, create new object
            if query_strategy.lower() == "batch_random":
                sample_index = np.random.choice(tuple(unqueried_index_set))
                unqueried_index_set.remove(sample_index)
                queried_index_set.add(sample_index)

                my_objective = MyObjective(train_set=train_set, target=target, num_class=num_class,
                                           train_idx=list(queried_index_set), val_idx=val_idx)
                # fair compare with incremental version
                study = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                            sampler=optuna.samplers.RandomSampler(), direction="minimize")
                study.optimize(my_objective, n_trials=(len(queried_index_set)-start_n_sample+1)*n_trial, callbacks=[my_objective.callback])
                model = my_objective.best_booster
                oof[len(queried_index_set) - 1, val_idx, :] = model.predict(train_set.iloc[val_idx],
                                                                            num_iteration=model.best_iteration)
            else:
                sample_index = query_index(model=my_objective.best_booster, train_set=train_set,
                                           unqueried_index_set=unqueried_index_set,
                                           query_strategy=query_strategy)
                unqueried_index_set.remove(sample_index)
                queried_index_set.add(sample_index)
                my_objective.train_idx = list(queried_index_set)
                study.optimize(my_objective, n_trials=n_trial, callbacks=[my_objective.callback])
                model = my_objective.best_booster
                oof[len(queried_index_set)-1, val_idx, :] = model.predict(train_set.iloc[val_idx], num_iteration=model.best_iteration)

    for i_oof in range(len(oof)):
        if i_oof < start_n_sample:
            continue
        pred_label = np.argmax(oof[i_oof], axis=1)
        accuracy = np.sum(pred_label == target) / len(target)
        result_pred[i_oof] = accuracy

    return result_pred


result_pred = np.zeros([n_run, end_n_sample])
for i_run in range(n_run):
    print("------ Round: {}/{}".format(i_run+1, n_run))
    result_pred[i_run] = run_cv(train_set, target_cf, len(set(target_cf)), n_sample_arr)

# print(result_pred)
# result_pred = np.delete(result_pred, list(range(start_n_sample)), axis=1)
print(result_pred)

# do not want dot in filenames
query_strategy = query_strategy.replace('.', '')
print("Saving result to ./Result/{}_result.csv".format(query_strategy))
savetxt("./Result/{}_result.csv".format(query_strategy), result_pred, delimiter=',')




