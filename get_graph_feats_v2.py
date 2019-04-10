import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from collections import Counter, defaultdict
from tqdm import tqdm
tqdm.pandas(tqdm)

from utils import read_data
from config import UTILITY
from sklearn.base import BaseEstimator, TransformerMixin


class GraphFeats(BaseEstimator, TransformerMixin):
    def __init__(self, use_chat=False, directed=False):
        self.use_chat = use_chat
        self.directed = directed
        self.neighbors_dict = defaultdict(set)

    def __build_neighbors(self, edges):
        for edge in tqdm(edges):
            a, b = edge
            self.neighbors_dict[a].add(b)
            if not self.directed:
                self.neighbors_dict[b].add(a)

    def __get_feats(self, edge):
        u, v = edge
        n1 = self.neighbors_dict[u]
        n2 = self.neighbors_dict[v]
        n1n2 = n1 & n2
        if len(n1n2) == 0:
            return 0, 0
        else:
            a = []
            b = []
            
            for w in list(n1n2):
                a.append(u in self.neighbors_dict[w])
                b.append(u in self.neighbors_dict[w])
            
        return sum(a), sum(b)
    
    def fit(self, X, y=None):
        if self.use_chat:
            X = X.loc[X.is_chat == 1, ["node1_id", "node2_id"]].values
        else:
            X = X[["node1_id", "node2_id"]].values
        self.__build_neighbors(X)
        return self

    def transform(self, X, y=None):
        X = X[["node1_id", "node2_id"]].values
        n = len(X)
        X_feats = np.zeros((n, 2), dtype=np.int32)
        for i in tqdm(range(n)):
            X_feats[i] = self.__get_feats(X[i])
        return X_feats


if __name__ == "__main__":
    train, test, users = read_data()

    print("Generating feature set 6")
    grf = GraphFeats(use_chat=True, directed=False)
    cvlist = list(StratifiedKFold(10, shuffle=True, random_state=12345786).split(train, train.is_chat))
    train_chat = cross_val_predict(grf, train, cv=cvlist, method='transform')
    test_chat = grf.fit(train).transform(test)
    np.save("train_gfeats_v8.npy", train_chat)
    np.save("test_gfeats_v8.npy", test_chat)
    del train_chat, test_chat
    gc.collect()

