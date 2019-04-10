import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from pathlib import Path
from collections import Counter
from tqdm import tqdm
from utils import read_data
from sklearn.base import BaseEstimator, TransformerMixin


class SwapChat(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        self.tr_dict = {}
        for recs in tqdm(X[["node1_id", "node2_id", "is_chat"]].values):
            self.tr_dict[(recs[0], recs[1])] = recs[2]
        return self
    
    def transform(self, X, y=None):
        X_out = -1*np.ones((len(X)))
        for i, rec in tqdm(enumerate(X[["node1_id", "node2_id"]].values)):
            X_out[i] = self.tr_dict.get((rec[1], rec[0]), -1)
        return X_out.reshape(-1,1)
    
    
if __name__=="__main__":
    train, test, users = read_data()
    cvlist =  list(StratifiedKFold(10, shuffle=True, random_state=12345786).split(train, train.is_chat))
    sw = SwapChat()
    x_train = cross_val_predict(sw, train, cv=cvlist, method="transform")
    x_test = sw.fit(train).transform(test)
    
    np.save("train_swap.npy", x_train)
    np.save("test_swap.npy", x_test)    