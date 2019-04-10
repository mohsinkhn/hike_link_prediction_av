import pandas as pd
import numpy as np

from config import SUBMISSIONS


if __name__=="__main__":
    sub1 = pd.read_csv(str(Path(SUBMISSIONS) / "sub_v6.csv"))
    sub2 = pd.read_csv(str(Path(SUBMISSIONS) / "sub_final_vt3.csv"))
    sub2["is_chat"] = gmean([sub2["is_chat"].values, sub1["is_chat"].values], axis=0)
    sub2.to_csv("sub_v9.csv", index=False)

    sub_v9 = pd.read_csv(str(Path(SUBMISSIONS) / "sub_v9.csv"))
    sub_nn1 = pd.read_csv(str(Path(SUBMISSIONS) / "sub_nnv1.csv"))

    sub_v10 = sub_v9[["id"]]
    sub_v10["is_chat"] = 0.5*sub_v9.is_chat + 0.5*sub_nn1.is_chat
    sub_v10.to_csv(str(Path(SUBMISSIONS) / "sub_v10.csv", index=False))

