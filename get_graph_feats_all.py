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
    def __init__(self, use_chat=False, directed=False, remove_self_nodes=False):
        self.use_chat = use_chat
        self.directed = directed
        self.neighbors_dict = defaultdict(set)
        self.remove_self_nodes = remove_self_nodes

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
        un = len(n1)
        vn = len(n2)
        cn = len(n1n2)  # common neighbors
        tn = len(n1 | n2)  # total neighbors
        if tn == 0:
            jc = 0
        else:
            jc = cn / tn  # jaccard
        pa = un * vn  # preferential attachment
        nd = cn / (1 + np.sqrt(pa))  # neighborhood distance
        n1n2_n = []
        if cn == 0:
            aa, ra = 0, 0
        else:
            for w in list(n1n2):
                n1n2_n.append(len(self.neighbors_dict[w]))
            aa = sum([1 / np.log(2 + lw) for lw in n1n2_n])
            ra = sum([1 / (1 + lw) for lw in n1n2_n])
        return un, vn, cn, tn, int(100 * jc), pa, int(100 * nd), int(100 * aa), int(100 * ra)

    def fit(self, X, y=None):
        if self.use_chat:
            X = X.loc[X.is_chat == 1, ["node1_id", "node2_id"]]
        else:
            X = X[["node1_id", "node2_id"]]
        if self.remove_self_nodes:
            X = X.loc[X.node1_id != X.node2_id].values
        else:
            X = X.values
        self.__build_neighbors(X)
        return self

    def transform(self, X, y=None):
        X = X[["node1_id", "node2_id"]].values
        n = len(X)
        X_feats = np.zeros((n, 9), dtype=np.int32)
        for i in tqdm(range(n)):
            X_feats[i] = self.__get_feats(X[i])
        return X_feats


if __name__ == "__main__":
    train, test, users = read_data()
    all_recs = pd.concat([train, test], axis=0)
    
    #print("Generating feature set 1")
    #grf = GraphFeats(use_chat=False, directed=False, remove_self_nodes=False)
    #grf.fit(all_recs)
    #train_contact = grf.transform(train)
    #test_contact = grf.transform(test)
    #np.save(UTILITY + "/train_gfeats_v1.npy", train_contact)
    #np.save(UTILITY + "/test_gfeats_v1.npy", test_contact)
    #del train_contact, test_contact
    #gc.collect()

    #print("Generating feature set 21")
    #grf = GraphFeats(use_chat=False, directed=False, remove_self_nodes=True)
    #grf.fit(all_recs)
    #train_contact = grf.transform(train)
    #test_contact = grf.transform(test)
    #np.save(UTILITY + "/train_gfeats_v21.npy", train_contact)
    #np.save(UTILITY + "/test_gfeats_v21.npy", test_contact)
    #del train_contact, test_contact
    #gc.collect()

    #print("Generating feature set 2")
    #grf = GraphFeats(use_chat=False, directed=True, remove_self_nodes=False)
    #grf.fit(all_recs)
    #train_contact = grf.transform(train)
    #test_contact = grf.transform(test)
    #np.save(UTILITY + "/train_gfeats_v2.npy", train_contact)
    #np.save(UTILITY + "/test_gfeats_v2.npy", test_contact)
    #del train_contact, test_contact
    #gc.collect()
    
    #print("Generating feature set 22")
    #grf = GraphFeats(use_chat=False, directed=True, remove_self_nodes=True)
    #grf.fit(all_recs)
    #train_contact = grf.transform(train)
    #test_contact = grf.transform(test)
    #np.save(UTILITY + "/train_gfeats_v22.npy", train_contact)
    #np.save(UTILITY + "/test_gfeats_v22.npy", test_contact)
    #del train_contact, test_contact
    #gc.collect()
    
    print("Generating feature set 3-- 5 folds")
    grf = GraphFeats(use_chat=True, directed=False, remove_self_nodes=True)
    cvlist = list(StratifiedKFold(5, shuffle=True, random_state=12345786).split(train, train.is_chat))
    train_chat = cross_val_predict(grf, train, cv=cvlist, method='transform')
    test_chat = grf.fit(train).transform(test)
    np.save(UTILITY + "/train_gfeats_v03.npy", train_chat)
    np.save(UTILITY + "/test_gfeats_v03.npy", test_chat)
    del train_chat, test_chat
    gc.collect()
    
    print("Generating feature set 3")
    grf = GraphFeats(use_chat=True, directed=False, remove_self_nodes=False)
    cvlist = list(StratifiedKFold(10, shuffle=True, random_state=12345786).split(train, train.is_chat))
    train_chat = cross_val_predict(grf, train, cv=cvlist, method='transform')
    test_chat = grf.fit(train).transform(test)
    np.save(UTILITY + "/train_gfeats_v3.npy", train_chat)
    np.save(UTILITY + "/test_gfeats_v3.npy", test_chat)
    del train_chat, test_chat
    gc.collect()

    print("Generating feature set 23")
    grf = GraphFeats(use_chat=True, directed=False, remove_self_nodes=True)
    cvlist = list(StratifiedKFold(20, shuffle=True, random_state=12345786).split(train, train.is_chat))
    train_chat = cross_val_predict(grf, train, cv=cvlist, method='transform')
    test_chat = grf.fit(train).transform(test)
    np.save(UTILITY + "/train_gfeats_v23.npy", train_chat)
    np.save(UTILITY + "/test_gfeats_v23.npy", test_chat)
    del train_contact, test_contact
    gc.collect()
    
    #print("Generating feature set 4")
    #grf = GraphFeats(use_chat=True, directed=True, remove_self_nodes=False)
    #cvlist = list(StratifiedKFold(10, shuffle=True, random_state=12345786).split(train, train.is_chat))
    #train_chat = cross_val_predict(grf, train, cv=cvlist, method='transform')
    #test_chat = grf.fit(train).transform(test)
    #np.save(UTILITY + "/train_gfeats_v4.npy", train_chat)
    #np.save(UTILITY + "/test_gfeats_v4.npy", test_chat)
    #del train_chat, test_chat
    #gc.collect()

    #print("Generating feature set 24")
    #grf = GraphFeats(use_chat=True, directed=True, remove_self_nodes=True)
    #cvlist = list(StratifiedKFold(20, shuffle=True, random_state=12345786).split(train, train.is_chat))
    #train_chat = cross_val_predict(grf, train, cv=cvlist, method='transform')
    #test_chat = grf.fit(train).transform(test)
    #np.save(UTILITY + "/train_gfeats_v24.npy", train_chat)
    #np.save(UTILITY + "/test_gfeats_v24.npy", test_chat)
    #del train_chat, test_chat
    #gc.collect()
