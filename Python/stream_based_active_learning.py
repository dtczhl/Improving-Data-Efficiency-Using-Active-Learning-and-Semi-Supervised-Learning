"""
    Stream-based active learning

    Supported strategy:
        query_by_disagreement: Query By Disagreement (QBD)
"""

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import optuna
from numpy import savetxt
import numpy as np
import os
import pandas as pd
from copy import deepcopy

import argparse
from argparse import RawTextHelpFormatter

from MyObjective import MyObjective
from Query_Strategy import query_index


# ------ Configurations ------

# query_by_disagreement

# take in as arguments
# query_strategy = "query_by_disagreement"

# do not change
n_sample_arr = list(range(3, 91))

# number of runs for each reduced number of samples
n_run = 1

# target distinctive numbers >
n_h_different_threshold = 2

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

# np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)

help_message = "Stream based Active Learning Sampling Method. " \
    "Supported Methods:\n" \
    "    query_by_disagreement\n"
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
    result_keep_size = np.zeros(n_fold * end_n_sample)
    result_keep_accuracy = np.zeros(n_fold * end_n_sample, dtype=float)

    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        unqueried_index_set = set(train_idx)
        queried_index_set = set()
        keep_queried_index_set = set()

        # randomly select instances for initialization
        for _ in range(start_n_sample):
            sample_index = np.random.choice(tuple(unqueried_index_set))
            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)
            keep_queried_index_set.add(sample_index)

        my_objective = MyObjective(train_set=train_set, target=target, num_class=num_class,
                                   train_idx=list(keep_queried_index_set), val_idx=val_idx)

        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.RandomSampler(), direction="minimize")
        study.optimize(my_objective, n_trials=n_trial, callbacks=[my_objective.callback])
        model = my_objective.best_booster
        oof[len(queried_index_set), val_idx, :] = model.predict(train_set.iloc[val_idx],
                                                                num_iteration=model.best_iteration)

        while len(queried_index_set) < end_n_sample:

            print("\t #sample: {}/{}".format(len(queried_index_set)+1, end_n_sample))

            if query_strategy.lower() == "query_by_disagreement":
                sample_index = np.random.choice(tuple(unqueried_index_set))
                unqueried_index_set.remove(sample_index)
                queried_index_set.add(sample_index)
                # will be removed if not satisfied
                keep_queried_index_set.add(sample_index)

                # print(my_objective.best_booster.predict(train_set.iloc[sample_index], num_iteration=my_objective.best_booster.best_iteration))

                # decision here
                objective_1, objective_2, objective_3, objective_4 \
                    = deepcopy(my_objective), deepcopy(my_objective), deepcopy(my_objective), deepcopy(my_objective)

                target_1, target_2, target_3, target_4 \
                    = np.copy(target), np.copy(target), np.copy(target), np.copy(target)
                target_1[sample_index], target_2[sample_index], target_3[sample_index], target_4[sample_index] \
                    = 0, 1, 2, 3

                objective_1.target, objective_2.target, objective_3.target, objective_4.target \
                    = target_1, target_2, target_3, target_4

                objective_1.train_idx, objective_2.train_idx, objective_3.train_idx, objective_4.train_idx \
                    = list(keep_queried_index_set), list(keep_queried_index_set), \
                      list(keep_queried_index_set), list(keep_queried_index_set)

                study_1 = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                              sampler=optuna.samplers.RandomSampler(), direction="minimize")
                study_1.optimize(objective_1, n_trials=n_trial, callbacks=[objective_1.callback])
                model_1 = objective_1.best_booster
                model_1_pred = np.argmax(np.squeeze(model_1.predict(train_set.iloc[sample_index])))
                # print(model_1.predict(train_set.iloc[sample_index]))

                study_2 = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                              sampler=optuna.samplers.RandomSampler(), direction="minimize")
                study_2.optimize(objective_2, n_trials=n_trial, callbacks=[objective_2.callback])
                model_2 = objective_2.best_booster
                model_2_pred = np.argmax(np.squeeze(model_2.predict(train_set.iloc[sample_index])))
                # print(model_2.predict(train_set.iloc[sample_index]))

                study_3 = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                              sampler=optuna.samplers.RandomSampler(), direction="minimize")
                study_3.optimize(objective_3, n_trials=n_trial, callbacks=[objective_3.callback])
                model_3 = objective_3.best_booster
                model_3_pred = np.argmax(np.squeeze(model_3.predict(train_set.iloc[sample_index])))
                # print(model_3.predict(train_set.iloc[sample_index]))

                study_4 = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                              sampler=optuna.samplers.RandomSampler(), direction="minimize")
                study_4.optimize(objective_4, n_trials=n_trial, callbacks=[objective_4.callback])
                model_4 = objective_4.best_booster
                model_4_pred = np.argmax(np.squeeze(model_4.predict(train_set.iloc[sample_index])))

                # print(model_4.predict(train_set.iloc[sample_index]))

                # print(target[sample_index], model_1_pred, model_2_pred, model_3_pred, model_4_pred)

                h_different_set = {model_1_pred, model_2_pred, model_3_pred, model_4_pred}

                if len(h_different_set) > n_h_different_threshold:
                    print("Discard")
                    keep_queried_index_set.remove(sample_index)
                else:
                    print("Keep")
                    my_objective.train_idx = list(keep_queried_index_set)
                    study.optimize(my_objective, n_trials=n_trial, callbacks=[my_objective.callback])

                model = my_objective.best_booster
                oof[len(queried_index_set) - 1, val_idx, :] = model.predict(train_set.iloc[val_idx])

                keep_size_pred_label = np.argmax(model.predict(train_set.iloc[val_idx]), axis=1)
                keep_accuracy = np.sum(keep_size_pred_label == target[val_idx]) / len(val_idx)

                print(len(keep_queried_index_set), keep_accuracy)

                result_keep_size[len(queried_index_set)-1 + (fold_ * end_n_sample)] = len(keep_queried_index_set)
                result_keep_accuracy[len(queried_index_set)-1 + (fold_ * end_n_sample)] = keep_accuracy

            else:
                print("Unknown {}".format(query_strategy))
                exit(-1)

    for i_oof in range(len(oof)):
        if i_oof < start_n_sample:
            continue
        pred_label = np.argmax(oof[i_oof], axis=1)
        accuracy = np.sum(pred_label == target) / len(target)
        result_pred[i_oof] = accuracy

    return result_pred, result_keep_size, result_keep_accuracy

result_pred = np.zeros([n_run, end_n_sample])
result_keep_size = np.zeros([n_run, n_fold*end_n_sample])
result_keep_accuracy = np.zeros([n_run, n_fold*end_n_sample], dtype=float)

for i_run in range(n_run):
    print("------ Round: {}/{}".format(i_run+1, n_run))
    result_pred[i_run], result_keep_size[i_run], result_keep_accuracy[i_run] \
        = run_cv(train_set, target_cf, len(set(target_cf)), n_sample_arr)

# print(result_pred)
# result_pred = np.delete(result_pred, list(range(start_n_sample)), axis=1)
print(result_pred)
print(result_keep_size)
print(result_keep_accuracy)

# do not want dot in filenames
query_strategy = query_strategy.replace('.', '')

print("Saving accuracy to ./Result/{}_result.csv".format(query_strategy))
savetxt("./Result/{}_result.csv".format(query_strategy), result_pred, delimiter=',')

print("Saving keep size to ./Result/{}_keep_size.csv".format(query_strategy))
savetxt("./Result/{}_keep_size.csv".format(query_strategy), result_keep_size, delimiter=',')

print("Saving keep accuracy size to ./Result/{}_keep_accuracy.csv".format(query_strategy))
savetxt("./Result/{}_keep_accuracy.csv".format(query_strategy), result_keep_accuracy, delimiter=',')




