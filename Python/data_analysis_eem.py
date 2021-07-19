"""
    Saving to CSV files

    EEM dataset
"""

import os

import pandas as pd
import numpy as np


# path to the data file
eem_df = pd.read_excel('../data/se_sp_sep_260.xlsx', sheet_name='Sheet2', engine='openpyxl')

# extract sp, se, and sep
sp_df = eem_df.filter(regex='SP[0-9]+')
se_df = eem_df.filter(regex='SE[0-9]+')
sep_df = eem_df.filter(regex='SEP[0-9]+')

feature_ex = eem_df["EX"].values.tolist()
feature_em = eem_df["Em"].values.tolist()

for i in range(len(feature_ex)):
    feature_ex[i] = feature_ex[i] * 1000 + feature_em[i]

np.savetxt("../data/Processed/eem_ex_em.csv", feature_ex, delimiter=",")


# transpose
sp_df = sp_df.transpose()
se_df = se_df.transpose()
sep_df = sep_df.transpose()

# add label
sp_df["Label"] = 0
se_df["Label"] = 1
sep_df["Label"] = 2

df = pd.concat([sp_df, se_df, sep_df], ignore_index=True)

df.to_csv("../data/Processed/eem_dataset.csv", index=False)

print(df)


