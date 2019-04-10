import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
from scipy.stats import gmean
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import Sequence
import keras
from keras.layers import Input, Dense, Embedding, concatenate, Dropout, Add, BatchNormalization, Flatten, Dot
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
from pathlib import Path
from tqdm import tqdm
np.random.seed(1001)

from utils import read_data


def index_df(df):
    inds = pd.DataFrame(index=range(8874627))
    df = pd.merge(inds, df, how="left", right_index=True, left_index=True).fillna(-3).values
    return df



def cross_val_predict(cvlist, te_len):
    y_test_all = []
    for i, (tr_inds, val_inds) in enumerate(cvlist):
        if i > 1:
            continue
        train_datagen = NodeDataGenerator(tr_inds, train.is_chat.values[tr_inds], shuffle=False,
                                          batch_size=10000, flag="train", node2vec=node2vec)
        val_datagen = NodeDataGenerator(val_inds, train.is_chat.values[val_inds], shuffle=False,
                                        batch_size=20000, flag="train", node2vec=node2vec)
        nnv1 = NNv1()
        callbacks = [ROC_AUC(val_datagen)]
        nnv1.model.fit_generator(train_datagen,
                             epochs=8,
                             use_multiprocessing=False, workers=16,
                             callbacks=callbacks)
        test_datagen = NodeDataGenerator(np.arange(0, te_len), np.zeros((te_len,)), shuffle=False,
                                         batch_size=10000, flag="test", node2vec=node2vec)
        y_test_preds = nnv1.model.predict_generator(test_datagen)
        y_test_all.append(y_test_preds.flatten())
        np.save("y_test_preds_nn_{}".format(i), y_test_preds.flatten())
    return y_test_all


class NodeDataGenerator(Sequence):
    def __init__(self, row_indices=None,
                       labels=None,
                       users=None,
                       batch_size=1024,
                       num_feats=13,
                       flag = "train",
                       shuffle=True,
                       node2vec=None):
        self.row_indices = row_indices
        self.num_feats = num_feats
        self.batch_size = batch_size
        self.indices = np.arange(0, len(row_indices))
        self.shuffle = shuffle
        self.labels = labels
        self.flag = flag
        self.node2vec = node2vec
        self.read_data()
        self.on_epoch_end()
        
    def read_data(self):
        self.X_nodes = np.load("{}_nodes.npy".format(self.flag), mmap_mode="r")
        self.X_base = np.load("{}_base_feats.npy".format(self.flag), mmap_mode="r")
        self.X_graph = np.load("{}_graph_feats.npy".format(self.flag), mmap_mode="r")
        
    def __len__(self):
        return int(np.ceil(len(self.row_indices)/self.batch_size))
   
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def __getitem__(self, idx):
        idxxx = self.indices[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_indices = self.row_indices[idxxx]

        X1 = self.X_base[batch_indices]
        X2 = self.X_graph[batch_indices]
        nodes = self.X_nodes[batch_indices]
        X3 = self.node2vec[nodes[:, 0]]
        X4 = self.node2vec[nodes[:, 1]]
        y = self.labels[idxxx]
        return [X1, X2, X3, X4], y.reshape(-1,1)


class ROC_AUC(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.y_val = np.vstack([y for x, y in validation_data])
    
    def on_epoch_end(self, epoch, logs={}):
        y_preds = self.model.predict_generator(self.validation_data).flatten()
        print("ROC AUC for this fold, is ", roc_auc_score(self.y_val, y_preds))
        
        
class NNv1():
    def __init__(self, embed_dim=32, num_feats=13, dim1=64):
        self.embed_dim = embed_dim
        self.num_feats = num_feats
        self.dim1 = dim1
        self.model = self.build_model()

    def build_model(self):
        inp1 = Input(shape=(56,))
        inp2 = Input(shape=(15,))
        inp3 = Input(shape=(32,))
        inp4 = Input(shape=(32,))
        dd = Dot(1)([inp3, inp4])
        x = concatenate([inp1, inp2, inp3, inp4, dd])
        x = BatchNormalization()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation="relu")(x)
        out = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=[inp1, inp2, inp3, inp4], outputs=out)
        opt = Adam(lr=0.001) #SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        model.summary()
        
        return model
    

if __name__=="__main__":

    train, test , users = read_data("data/")

    node2vec = pd.read_csv("hike_node2vec.emb", sep=" ", skiprows=1,
                       names=["node_id"] + ["emb_{}".format(i) for i in range(32)], header=None).set_index("node_id")
    node2vec = index_df(node2vec)
    cvlist = list(StratifiedKFold(5, random_state=12345786).split(train.is_chat, train.is_chat))
    te_len = len(test)
    y_test_all = cross_val_predict(cvlist, te_len)


    ytest1 = np.load("y_test_preds_nn_0.npy")
    ytest2 = np.load("y_test_preds_nn_1.npy")
    y_test_all = [ytest1, ytest2]


    y_test_preds = gmean(y_test_all, axis=0)

    sub = test[["id"]]
    sub["is_chat"] = y_test_preds
    sub.to_csv(str(Path(SUBMISSIONS) / "sub_nnv1.csv"), index=False)

