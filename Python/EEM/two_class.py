"""
    sp vs sep, se vs sep, and sp + se vs sep
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


models = [
    KNeighborsClassifier(1),
    KNeighborsClassifier(2),
    KNeighborsClassifier(3),
    SVC(kernel='linear', C=0.02),
    SVC(kernel='poly'),
    SVC(kernel='rbf'),
    SVC(kernel='sigmoid'),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(hidden_layer_sizes=[100, 100]),
    AdaBoostClassifier(),
    GaussianNB()
]

model_names = [
    'Nearest (1) Neighbors',
    'Nearest (2) Neighbors',
    'Nearest (3) Neighbors',
    'SVC Kernel=linear',
    'SVC Kernel=poly',
    'SVC Kernel=rbf',
    'SVC Kernel=sigmoid',
    'Gaussian Process',
    'Decision Tree',
    'Random Forest',
    'MLPClassifier 100-100',
    'AdaBoost',
    'Naive Bayes'
]


# number of runs
n_run = 100

# number of folds for cross-validation.
n_fold = 5

# path to the data file
eem_df = pd.read_excel('../data/se_sp_sep_260.xlsx', sheet_name='Sheet2', engine='openpyxl')

# save results
save_file = './EEM/eem_binary_result.csv'
f = open(save_file, "w")
f.write("Model,sp-sep,se-sep,sp+se-sep\n")

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
se_df["Label"] = 1
sep_df["Label"] = 2

# complete datasets for sp-sep and se-sep
sp_sep_df = pd.concat([sp_df, sep_df], ignore_index=True)
se_sep_df = pd.concat([se_df, sep_df], ignore_index=True)

# sp + se vs sep
sp_df["Label"] = -1
se_df["Label"] = -1
sp_se_sep_df = pd.concat([sp_df, se_df, sep_df], ignore_index=True)

# change column names
column_names = sp_sep_df.columns
new_column_names = ['Label' if x == 'Label' else 'Feature_' + str(x+1) for x in column_names]
sp_sep_df.columns = new_column_names
se_sep_df.columns = new_column_names
sp_se_sep_df.columns = new_column_names

# training data for sp vs sep
train_set_sp_sep = sp_sep_df.drop(["Label"], axis=1)
target_sp_sep = sp_sep_df["Label"]

# training data for se vs sep
train_set_se_sep = se_sep_df.drop(["Label"], axis=1)
target_se_sep = se_sep_df["Label"]

# training data for sp+se vs sep
train_set_sp_se_sep = sp_se_sep_df.drop(["Label"], axis=1)
target_sp_se_sep = sp_se_sep_df["Label"]

# data preprocessing
#  normalization
stand = preprocessing.StandardScaler()
data_sp_sep = stand.fit_transform(train_set_sp_sep)
data_sp_sep = pd.DataFrame(data_sp_sep)
data_sp_sep.columns = train_set_sp_sep.columns

data_se_sep = stand.fit_transform(train_set_se_sep)
data_se_sep = pd.DataFrame(data_se_sep)
data_se_sep.columns = train_set_se_sep.columns

data_sp_se_sep = stand.fit_transform(train_set_sp_se_sep)
data_sp_se_sep = pd.DataFrame(data_sp_se_sep)
data_sp_se_sep.columns = train_set_sp_se_sep.columns

# re-labeling
le = preprocessing.LabelEncoder()
target_cf_sp_sep = le.fit_transform(target_sp_sep)
target_cf_se_sep = le.fit_transform(target_se_sep)
target_cf_sp_se_sep = le.fit_transform(target_sp_se_sep)


def run_cv(model, train_set, target):

    folds = KFold(n_splits=n_fold, shuffle=True)

    accuracy = 0
    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        model.fit(train_set.iloc[train_idx], target[train_idx])
        pred = model.predict(train_set.iloc[val_idx])
        pred_label = 1 * (pred >= 0.5)
        accuracy += np.sum(pred_label == target[val_idx]) / len(target[val_idx]) / n_fold

    return accuracy


for i_model in range(len(models)):
    model = models[i_model]

    result_sp = 0
    result_se = 0
    result_sp_se = 0
    for i_run in range(n_run):

        result_sp += run_cv(model, data_sp_sep, target_cf_sp_sep) / n_run
        result_se += run_cv(model, data_se_sep, target_cf_se_sep) / n_run
        result_sp_se += run_cv(model, data_sp_se_sep, target_cf_sp_se_sep) / n_run

    print("{}, {:.2f}, {:.2f}, {:.2f}".format(model_names[i_model], result_sp, result_se, result_sp_se))
    f.write("{},{:.2f},{:.2f},{:.2f}\n".format(model_names[i_model], result_sp, result_se, result_sp_se))

f.flush()
f.close()

