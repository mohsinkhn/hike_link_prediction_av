import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from pathlib import Path
from collections import Counter
import networkit as nt
import networkx as nx
from tqdm import tqdm

from utils import read_data, merge_features


if __name__=="__main__":
    train, test, users = read_data("data")

    feats = ["f{}_x".format(i) for i in range(1, 13)] + ["f{}_y".format(i) for i in range(1, 13)]

    train = merge_features(train, users)

    scaler = QuantileTransformer(output_distribution="normal")
    onehot = OneHotEncoder(categories="auto", sparse=False)

    #onehot.fit(train[["f13_x", "f13_y"]])

    tr_all = np.hstack((train[feats].values/31, onehot.fit_transform(train[["f13_x", "f13_y"]])))

    np.save("train_base_feats.npy", tr_all)

    test = merge_features(test, users)

    te_all = np.hstack((test[feats].values/31, onehot.fit_transform(test[["f13_x", "f13_y"]])))

    np.save("test_base_feats.npy", te_all)


