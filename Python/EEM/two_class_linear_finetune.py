import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC


# number of runs for each reduced number of samples
n_run = 10

# number of folds for cross-validation. !Do not change
n_fold = 5

# path to the data file
eem_df = pd.read_excel('../../data/se_sp_sep_260.xlsx', sheet_name='Sheet2', engine='openpyxl')

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

# change column names
column_names = sp_sep_df.columns
new_column_names = ['Label' if x == 'Label' else 'Feature_' + str(x+1) for x in column_names]
sp_sep_df.columns = new_column_names
se_sep_df.columns = new_column_names

# training data for sp vs sep
train_set_sp_sep = sp_sep_df.drop(["Label"], axis=1)
target_sp_sep = sp_sep_df["Label"]

# training data for se vs sep
train_set_se_sep = se_sep_df.drop(["Label"], axis=1)
target_se_sep = se_sep_df["Label"]

# data preprocessing
#  normalization
stand = preprocessing.StandardScaler()
data_sp_sep = stand.fit_transform(train_set_sp_sep)
data_sp_sep = pd.DataFrame(data_sp_sep)
data_sp_sep.columns = train_set_sp_sep.columns

data_se_sep = stand.fit_transform(train_set_se_sep)
data_se_sep = pd.DataFrame(data_se_sep)
data_se_sep.columns = train_set_se_sep.columns

# re-labeling
le = preprocessing.LabelEncoder()
target_cf_sp_sep = le.fit_transform(target_sp_sep)
target_cf_se_sep = le.fit_transform(target_se_sep)


C_s = np.logspace(-100, 100, 10000)

scores = list()

svc = SVC(kernel='linear')

clf = GridSearchCV(estimator=svc, param_grid=dict(C=C_s), n_jobs=10)

clf.fit(data_se_sep, target_se_sep)

print(clf.best_score_)
print(clf.best_estimator_.C)
