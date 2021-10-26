import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold

from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

import warnings

models = [
    LogisticRegression(solver='newton-cg', multi_class='multinomial'),
    BernoulliNB(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    ExtraTreesClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
    LinearDiscriminantAnalysis(),
    LinearSVC(multi_class='crammer_singer'),
    MLPClassifier(),
    NearestCentroid(),
    QuadraticDiscriminantAnalysis(),
    RandomForestClassifier(),
    RidgeClassifier()
]

model_names = [
    'LogisticRegression',
    'Multivariate Bernoulli',
    'Decision Tree',
    'Extra Tree',
    'Extra Trees',
    'Gaussian Naive Bayes',
    'K-nearest Neighbor',
    'Linear Discriminant Analysis',
    'Linear Support Vector',
    'Multi-layer Perceptron',
    'Nearest Centroid',
    'Quadratic Discriminant Analysis',
    'Random Forest',
    'Ridge Regression'
]


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# number of runs
n_run = 20

# number of folds for cross-validation.
n_fold = 5

# path to the data file
eem_df = pd.read_excel('../../data/2ET_40_trim20_L.xlsx', sheet_name='Sheet1', engine='openpyxl')

# save results
save_file = './multi_class_result.csv'
f = open(save_file, "w")
f.write("Model,Accuracy\n")

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
data_all = pd.DataFrame(data_all)
data_all.columns = train_set_all.columns

# re-labeling
le = preprocessing.LabelEncoder()
target_cf_all = le.fit_transform(target_all)


def run_cv(model, train_set, target):

    folds = KFold(n_splits=n_fold, shuffle=True)

    accuracy = 0
    for fold_, (train_idx, val_idx) in enumerate(folds.split(train_set.values, target)):

        model.fit(train_set.iloc[train_idx], target[train_idx])
        pred = model.predict(train_set.iloc[val_idx])

        accuracy += np.sum(pred == target[val_idx]) / len(target[val_idx]) / n_fold

    return accuracy


for i_model in range(len(models)):
    model = models[i_model]

    result_all = 0
    for i_run in range(n_run):
        result_all += run_cv(model, data_all, target_cf_all) / n_run

    print("{}, {:.2f}".format(model_names[i_model], result_all))
    f.write("{},{:.2f}\n".format(model_names[i_model], result_all))

f.flush()
f.close()
