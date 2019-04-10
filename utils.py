import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from tqdm import tqdm
from pathlib import Path

from config import *


def read_data():
    dtypes1 = {
               "node1_id": np.int32,
               "node2_id": np.int32
            }
            
    dtypes2 = {"f{}".format(i):np.uint8 for i in range(1, 14)}
    dtypes2["node_id"] = np.int32
                    
    root_path = Path(ROOT)
    train = pd.read_csv(str(root_path / "train.csv"), dtype=dtypes1)
    test = pd.read_csv(str(root_path / "test.csv"), dtype=dtypes1)
    users = pd.read_csv(str(root_path / "user_features.csv"), dtype=dtypes2)
    return train, test, users


def merge_features(df, users):
    df = pd.merge(df, users, left_on="node1_id", right_on="node_id", how="left")
    del df["node_id"]
    df = pd.merge(df, users, left_on="node2_id", right_on="node_id", how="left")
    del df["node_id"]
    return df


class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col=None, thresh=0, smooth=False, alpha=0.1):
        self.thresh = thresh
        self.smooth = smooth
        self.alpha = alpha
        self.col = col

    def fit(self, X, y=None):
        cnts = Counter(X["node1_id"].values)
        cnts.update(X["node2_id"].values)

        sums = Counter(X.loc[X["is_chat"] == 1, "node1_id"].values)
        sums.update(X.loc[X["is_chat"] == 1, "node2_id"].values)

        self.means = {k: sums.get(k, v)/v for k, v in tqdm(cnts.items()) if v >= self.thresh}
        
    def transform(self, X, y=None):
        return X[self.col].map(self.means)
