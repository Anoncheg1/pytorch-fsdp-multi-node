#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging


# DATASET_PATH = "/mnt/datasets/DataSetAI/landmark-retrieval-2020/"
DATASET_PATH = "/datasets/DataSetAI/landmark-retrieval-2020/"

def calc_paths(ldir, land_id, max_count=-1):
    # -- init array
    path = land_id[0]
    l = len(f"{ldir}/{path[0]}/{path[1]}/{path[2]}/{path}.jpg")
    paths = np.chararray(land_id.shape[0], itemsize=l)
    # -- create pathes from id column from train.csv file
    for i in range(land_id.shape[0]):
        path = land_id[i]
        paths[i] = f"{ldir}/{path[0]}/{path[1]}/{path[2]}/{path}.jpg"
    return paths

def get_dataset(get_tail=None):
    # -- load dataset to pandas
    # -- profile -<-
    # import time
    # start_time = time.time()
    # --  ->-
    df = pd.read_csv(DATASET_PATH + "train.csv")  # count 1580470 # id  landmark_id
    paths = calc_paths(DATASET_PATH + "train", df["id"].to_numpy())
    df['path'] = paths

    df = df[~ df.path.isna()]  # select records with "path" column
    # -- speed up loading -/-
    # df.to_pickle('landmartk.pkl')
    # df = pd.read_pickle('landmartk.pkl')
    # -/-
    # -- counts
    counts_map = dict(df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['counts'] = df['landmark_id'].map(counts_map)
    # -- debug, select several -- / --
    # if get_tail is not None:
    #     # - select classes where we have enough examples
    #     df = df[df.counts > 42]
    #     df = df[df.counts < 50]
    #     df = df.tail(get_tail)
    #     logging.info("df.describe.to_string(): " + df.describe().to_string())
    #     counts_map = dict(df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    #     df['counts'] = df['landmark_id'].map(counts_map)

    # -- / --
    uniques = df['landmark_id'].unique() # unique classes
    OUTPUT_SIZE = len(uniques)
    df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques))))) # scale landmark_id to 0-
    logging.info("df.shape: " + str(df.shape))
    logging.info(df.describe().to_string())

    image_paths, labels = df.path.to_numpy(), df.label.to_numpy()

    x_train, x_valid, y_train, y_valid = train_test_split(image_paths, labels, test_size=0.20, stratify=labels)
    print("x_train, x_valid, y_train, y_valid", x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
    # x_train, y_train = df.path.to_numpy(), df.label.to_numpy()
    # print("--- %s seconds ---" % (time.time() - start_time))
    return x_train, x_valid, y_train, y_valid, OUTPUT_SIZE

# test
if __name__ == '__main__':
    res = get_dataset()
    print("len(res)", len(res))
    for i, data in enumerate(res):
        if isinstance(data, np.ndarray):
            print("i, data", i, data.shape)
