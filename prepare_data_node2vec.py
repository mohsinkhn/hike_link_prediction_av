import pandas as pd
import numpy as np
from pathlib import Path

from utils import read_data
from config import UTILITY


if __name__=="__main__":
    train, test, users = read_data()

    train_oof = np.load(str(Path(UTILITY) / "train_oof_preds.npy"))
    test_oof = np.load(str(Path(UTILITY) / "test_oof_preds.npy"))

    train["preds"] = train_oof
    test["preds"] = test_oof

    thresh = 0.001
    train.loc[train.preds > thresh, "is_chat"].sum(), train.loc[train.preds > thresh, "is_chat"].count()

    t1 = train.loc[train.preds > thresh, ["node1_id", "node2_id", "preds"]]
    t2 = test.loc[test.preds > thresh, ["node1_id", "node2_id", "preds"]]

    t_all = pd.concat([t1, t2], axis=0)
    print(t_all.shape)
    t_all.to_csv(str(Path(UTILITY) / "weighted_nodes.graph"), sep=" ", header=None, index=False)
