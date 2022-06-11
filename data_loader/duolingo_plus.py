import os

import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from models.utils import match_seq_len_for_SEKT


DATASET_DIR = "data/"


class DUOLINGO_plus(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, "train_data.csv"
        )

        self.q_seqs, self.s_seqs, self.rq_seqs, self.rs_seqs, self.q_list, \
            self.s_list, self.u_list, self.q2idx, self.s2idx, self.u2idx = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_s = self.s_list.shape[0]

        if seq_len:
            self.q_seqs, self.s_seqs, self.rq_seqs, self.rs_seqs = \
                match_seq_len_for_SEKT(self.q_seqs, self.s_seqs, self.rq_seqs, self.rs_seqs, seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.s_seqs[index], self.rq_seqs[index], self.rs_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path).dropna(subset=["pos","sen_pos"])\
            .drop_duplicates(subset=["days", "pos","sen_pos"])\
            .sort_values(by=["days"])

        u_list = np.unique(df["user"].values)
        q_list = np.unique(df["pos"].values)
        s_list = np.unique(df["sen_pos"].values)

        u2idx = {u: idx for idx, u in enumerate(u_list)} #인덱스 번호지정
        q2idx = {q: idx for idx, q in enumerate(q_list)} #인덱스 번호지정
        s2idx = {s: idx for idx, s in enumerate(s_list)}

        q_seqs = []
        s_seqs = []
        rq_seqs = []
        rs_seqs = []

        for u in u_list:
            df_u = df[df["user"] == u]
            q_seq = np.array([q2idx[q] for q in df_u["pos"]])
            s_seq = np.array([s2idx[s] for s in df_u["sen_pos"]])
            rq_seq = df_u["correct"].values
            rs_seq = df_u["sen_correct"].values

            q_seqs.append(q_seq)
            s_seqs.append(s_seq)
            rq_seqs.append(rq_seq)
            rs_seqs.append(rs_seq)
  
        return q_seqs, s_seqs, rq_seqs, rs_seqs, q_list, s_list, u_list, q2idx, s2idx, u2idx
