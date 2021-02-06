import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import optuna
from numpy import savetxt
import os

from MyObjective import MyObjective


# ------ Configurations ------

# do not change
n_sample_arr = list(range(3, 91))

# number of runs for each reduced number of samples
n_run = 20

# optuna number of trails
n_trial = 500

# data directory
dire = "../data/PAW FTIR data/"

# number of folds for cross-validation. !Do not change
n_fold = 5

# --- End of Configurations ---

np.random.seed(123)
optuna.logging.set_verbosity(optuna.logging.FATAL)

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


def run_cv(train_set, target, num_class, n_sample):

    folds = KFold(n_splits=n_fold, shuffle=True)
    oof = np.zeros([len(train_set), num_class])
    feature_importance_df = pd.DataFrame()

    # below is to split the entire dataset into train and test sets; Train on train set and evaluate on test set
    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        train_X, train_Y = train_set.iloc[train_idx], target[train_idx]
        test_X, test_Y = train_set.iloc[val_idx], target[val_idx]

        unqueried_index_set = set(train_idx)
        queried_index_set = set()

        # pool for training
        train_X_pool = train_X[0:0]
        train_y_pool = []

        # randomly select 3 instances for initialization
        for _ in range(3):
            sample_index = np.random.choice(tuple(unqueried_index_set))
            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)
            train_X_pool = train_X_pool.append(train_set.iloc[sample_index])
            train_y_pool.append(target[sample_index])

        print(list(queried_index_set))
        exit()

        my_objective = MyObjective(train_set=train_set, target=target, num_class=num_class,
                                   train_idx=list(queried_index_set), val_idx=val_idx)

        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), direction="minimize")
        study.optimize(my_objective, n_trials=n_trial, callbacks=[my_objective.callback])


        exit()

        while len(train_X_pool) < n_sample:
            # select one more instance

            print(len(train_X_pool))

            sample_index = np.random.choice(tuple(unqueried_index_set))
            unqueried_index_set.remove(sample_index)
            queried_index_set.add(sample_index)
            train_X_pool = train_X_pool.append(train_set.iloc[sample_index])
            train_y_pool.append(target[sample_index])

            train_data = lgb.Dataset(train_X_pool, label=train_y_pool)



        model = booster.get_best_booster()
        oof[val_idx, :] = model.predict(test_X, num_iteration=model.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train_set.columns
        fold_importance_df["importance"] = model.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    return feature_importance_df, oof


# print("Full dataset: {}".format(train_set.shape))
result_pred = np.zeros([len(n_sample_arr), n_run])
for i_run in range(n_run):
    result_pred[i_run] = run_cv(train_set, target_cf, len(set(target_cf)))


for i_train_sample in range(len(n_train_sample_arr)):
    n_train_sample = n_train_sample_arr[i_train_sample]

    for i_run in range(n_run):

        print("\n----- #samples: {}, #run: {}/{}".format(n_train_sample, i_run+1, n_run))

        feature_importance_df_cf, oof_cf = run_cv(train_set, target_cf, len(set(target_cf)), n_train_sample)
        pred = np.zeros([len(oof_cf)])
        for i in range(len(oof_cf)):
            pred[i] = np.argmax(oof_cf[i])
        cm = confusion_matrix(target_cf, pred)
        total_pred_correct = sum(cm.diagonal())/np.sum(cm)

        result_pred[i_train_sample][i_run] = total_pred_correct

print(result_pred)

savetxt("result_pred.csv", result_pred, delimiter=',')





