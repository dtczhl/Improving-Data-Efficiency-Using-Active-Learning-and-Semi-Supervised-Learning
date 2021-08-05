
import os

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans


# path to the data file
eem_df = pd.read_excel('../data/se_sp_sep_260.xlsx', sheet_name='Sheet2', engine='openpyxl')

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
se_df["Label"] = 0
sep_df["Label"] = 1

eem_new_df = pd.concat([sp_df, se_df, sep_df], ignore_index=True)

# change column names
column_names = eem_new_df.columns
new_column_names = ['Label' if x == 'Label' else 'Feature_' + str(x+1) for x in column_names]
eem_new_df.columns = new_column_names

train_set = eem_new_df.drop(["Label"], axis=1)
target = eem_new_df["Label"]

stand = preprocessing.StandardScaler()
data = stand.fit_transform(train_set)
data = pd.DataFrame(data)
data.columns = train_set.columns
train_set = data

le = preprocessing.LabelEncoder()
target_cf = le.fit_transform(target)

num_class = len(set(target_cf))

k_mean = KMeans(n_clusters=2, random_state=0, init='random').fit(train_set)

print(k_mean.labels_)
print(target_cf)