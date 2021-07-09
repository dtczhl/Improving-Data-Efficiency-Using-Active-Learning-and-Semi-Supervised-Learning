"""
    Saving to CSV files

    EEM dataset
"""

import os

import pandas as pd


# path to the data file
eem_df = pd.read_excel('../data/se_sp_sep_260.xlsx', sheet_name='Sheet2', engine='openpyxl')

# extract sp, se, and sep
sp_df = eem_df.filter(regex='SP[0-9]+')
se_df = eem_df.filter(regex='SE[0-9]+')
sep_df = eem_df.filter(regex='SEP[0-9]+')

feature_names = eem_df["Em"].values.tolist()
feature_names.append("Label")

# transpose
sp_df = sp_df.transpose()
se_df = se_df.transpose()
sep_df = sep_df.transpose()

# add label
sp_df["Label"] = 0
se_df["Label"] = 1
sep_df["Label"] = 2

df = pd.concat([sp_df, se_df, sep_df], ignore_index=True)

df.columns = feature_names

df.to_csv("../data/Processed/eem_dataset.csv", index=False)

print(df)


