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
from configs import UTILITY


if __name__=="__main__":
    train, test, users = read_data("data")
    tr_all = train[["node1_id", "node2_id"]].values
    np.save(str(Path(UTILITY) / "train_nodes.npy"), tr_all)

    te_all = test[["node1_id", "node2_id"]].values
    np.save(str(Path(UTILITY) / "test_nodes.npy"), te_all)

