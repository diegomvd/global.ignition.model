import pandas as pd
import numpy as np

ignition_df = pd.read_csv("./select_variables_fire.csv")

ignition_df = ignition_df.drop("Unnamed: 0", axis="columns")
print(ignition_df)

rng = np.random.default_rng()
id_array = ignition_df.index.to_numpy()
rng.shuffle(id_array)

n_folds = 10
splits_id = np.array_split(id_array,n_folds)

for i in range(n_folds):
    test = ignition_df.iloc[splits_id[i]]
    train = ignition_df.drop(splits_id[i], axis = 0)
    test.to_csv("./outer_folds/outcvloop_test_fold_{}.csv".format(i+1) ,index=False)
    train.to_csv("./outer_folds/outcvloop_train_fold_{}.csv".format(i+1) ,index=False)