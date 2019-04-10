import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import Counter
from tqdm import tqdm

from utils import read_data

def merge_features(df, users):
    df = pd.merge(df, users, left_on="node1_id", right_on="node_id", how="left")
    del df["node_id"]
    df = pd.merge(df, users, left_on="node2_id", right_on="node_id", how="left")
    del df["node_id"]
    return df


def merge_graph_feats(df, flag="train"):
    featsg = ["uv", "nv", "cn", "tn", "jc", "pa", "nd", "aa", "ra"]
    featsg2 = [f+"_2" for f in featsg]
    featsg3 = [f+"_3" for f in featsg]
    featsg4 = [f+"_4" for f in featsg]
    df_contact = pd.DataFrame(np.load("{}_graph_feats3.npy".format(flag)), columns=featsg)
    df_chat = pd.DataFrame(np.load("{}_graph_feats4.npy".format(flag)), columns=featsg2)
    df_chat3 = pd.DataFrame(np.load("{}_graph_feats5.npy".format(flag)), columns=featsg3)
    df_chat4 = pd.DataFrame(np.load("{}_graph_feats6.npy".format(flag)), columns=featsg4)
    df_swap = pd.DataFrame(np.load("{}_swap.npy".format(flag)), columns=["swap_feat"])
    df_all = pd.concat([df, df_contact, df_chat, df_chat3, df_chat4, df_swap], axis=1)
    df_all["f12_xy"] = df_all["f12_x"]/df_all["f12_y"]
    return df_all


if __name__ == "__main__":
    print("Reading data")
    train, test, users = read_data()
    
    print("Merging user activity features")
    train = merge_features(train, users)
    test = merge_features(test, users)

    print("merging grpah and extra features")
    train = merge_graph_feats(train, "train")
    test = merge_graph_feats(test, "test")
    
    feats1 = ["f{}_x".format(i) for i in range(1, 14)]
    feats2 = ["f{}_y".format(i) for i in range(1, 14)]
    featsg = ["uv", "nv", "cn", "tn", "jc", "pa", "nd", "aa", "ra"]
    featsg4 = [f+"_4" for f in featsg]
    feats2g_sel = ["uv_2", "nv_2", "cn_2", "tn_2"]
    feats3g_sel = ["uv_3", "nv_3", "cn_3", "tn_3"]

    all_feats = feats1+feats2+featsg + feats2g_sel + feats3g_sel + featsg4+ ["f12_xy", "swap_feat"]

    X = train[all_feats]
    y = train["is_chat"]
    
    X_test = test[all_feats]
    cvlist =  list(StratifiedKFold(5, shuffle=True, random_state=12345786).split(train, train.is_chat))
    model = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=200000, learning_rate=0.1, num_leaves=8,
                           subsample=0.9, colsample_bytree=0.9, seed=1, min_child_samples=60, n_jobs=-1,
                          reg_alpha=0.1, reg_lambda=10, min_child_weight=1)

    y_preds_lgb = np.zeros((len(y)))
    test_preds_allfolds = []
    for i, (tr_idx, val_idx) in enumerate(cvlist):
        X_dev, y_dev = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model.fit(X_dev, y_dev, eval_set=[(X_val, y_val)], eval_metric="auc", verbose=50, early_stopping_rounds=400,
             categorical_feature=['f13_x', 'f13_y'])
        val_preds = model.predict_proba(X_val)[:, 1]
        test_preds = model.predict_proba(X_test)[:, 1]
        test_preds_allfolds.append(test_preds)
        y_preds_lgb[val_idx] = val_preds
        print("Score for fold {} is {}".format(i, roc_auc_score(y_val, val_preds)))
    #break
    print("Overall Score for oof predictions ", roc_auc_score(y, y_preds_lgb))
    #model = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=10000, learning_rate=0.1, num_leaves=8,
    #                       subsample=0.9, colsample_bytree=0.9, seed=1, min_child_samples=60, n_jobs=-1,
    #                      reg_alpha=0.1, reg_lambda=10, min_child_weight=1)    
    #model.fit(X)
    from scipy.stats import gmean
    y_test_preds = gmean(test_preds_allfolds, axis=0)
    sub = test[['id']]
    sub["is_chat"] = y_test_preds
    sub.to_csv("sub_v5.csv", index=False)
                       
                            