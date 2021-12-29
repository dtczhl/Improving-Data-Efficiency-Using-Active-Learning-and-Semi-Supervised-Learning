"""
    K-means for plasma
"""

import os
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans



## path to the data file
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

# re-labeling
le = preprocessing.LabelEncoder()
target_cf = le.fit_transform(target_all)

num_class = len(set(target_cf))

k_mean = KMeans(n_clusters=4, random_state=0, init='random').fit(train_set)


print(k_mean.labels_)
print(target_cf)



