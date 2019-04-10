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
import networkit as nt
import networkx as nx
from tqdm import tqdm


from utils import read_data
from config import UTILITY, SUBMISSIONS


if __name__=="__main__":

    train, test, users = read_data("data")

    featsg = ["uv", "nv", "cn", "tn", "jc", "pa", "nd", "aa", "ra"]
    featsg2 = [f+"_2" for f in featsg]
    featsg3 = [f+"_3" for f in featsg]
    featsg4 = [f+"_4" for f in featsg]
    featsg5 = [f+"_5" for f in featsg]
    featsg6 = ["upath", "vpath"]
    train_contact = pd.DataFrame(np.load(str(Path(UTILITY) / "train_gfeats_v1.npy")), columns=featsg)
    train_chat = pd.DataFrame(np.load(str(Path(UTILITY) / "train_gfeats_v03.npy")), columns=featsg2)
    train_chat3 = pd.DataFrame(np.load(str(Path(UTILITY) / "train_gfeats_v2.npy")), columns=featsg3)
    train_chat5 = pd.DataFrame(np.load(str(Path(UTILITY) / "train_gfeats_v8.npy")), columns=featsg6)

    train_swap = pd.DataFrame(np.load(str(Path(UTILITY) / "train_swap.npy")), columns=["swap_feat"])

    tr_all = pd.concat([train, train_contact[["uv", "nv", "cn", "jc", "aa", "ra"]],
                    train_chat[["uv_2", "nv_2", "cn_2"]], train_chat3[["uv_3", "nv_3", "cn_3"]], train_chat5, train_swap], axis=1)

    feats = ["uv", "nv", "cn", "jc", "aa", "ra"] + ["uv_2", "nv_2", "cn_2"] + ["uv_3", "nv_3", "cn_3"] + ["upath", "vpath", "swap_feat"]

    scaler = QuantileTransformer(output_distribution="normal")

    tr_all = scaler.fit_transform(tr_all[feats])

    np.save(str(Path(UTILITY) / "train_graph_feats.npy"), tr_all)

    test_contact = pd.DataFrame(np.load(str(Path(UTILITY) / "test_gfeats_v1.npy")), columns=featsg)
    test_chat = pd.DataFrame(np.load(str(Path(UTILITY) / "test_gfeats_v03.npy")), columns=featsg2)
    test_chat3 = pd.DataFrame(np.load(str(Path(UTILITY) / "test_gfeats_v2.npy")), columns=featsg3)
    test_chat5 = pd.DataFrame(np.load(str(Path(UTILITY) / "test_gfeats_v8.npy")), columns=featsg6)

    test_swap = pd.DataFrame(np.load("test_swap.npy"), columns=["swap_feat"])

    te_all = pd.concat([test, test_contact[["uv", "nv", "cn", "jc", "aa", "ra"]],
                    test_chat[["uv_2", "nv_2", "cn_2"]], test_chat3[["uv_3", "nv_3", "cn_3"]], test_chat5, test_swap], axis=1)

    te_all = scaler.transform(te_all[feats])

    np.save(str(Path(UTILITY) / "test_graph_feats.npy"), te_all)
