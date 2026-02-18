import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout, Sequential, ReLU

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.emb_path = emb_path

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        lstm_input_dim = self.emb_size
        if self.emb_path:
            pretrained_weight = np.load(self.emb_path)
            pretrained_weight = torch.from_numpy(pretrained_weight).float()
            actual_pretrained_dim = pretrained_weight.shape[1]
            self.pretrained_emb = Embedding.from_pretrained(pretrained_weight,freeze=True)
            self.emb_projection = Sequential(
                Linear(actual_pretrained_dim, self.emb_size),
                ReLU(),
                Dropout(dropout)
            )
            self.gate_layer = Linear(self.emb_size * 2, self.emb_size)
        else:
            print("emb_path==\"\"")

        self.lstm_layer = LSTM(lstm_input_dim, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        

    def forward(self, q, r, s=None):
        # print(f"q.shape is {q.shape}")
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        # print(f"xemb.shape is {xemb.shape}")
        if self.emb_path:
            if s is None:
                raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
            code_emb = self.pretrained_emb(s)
            code_features = self.emb_projection(code_emb)

            combined = torch.cat([xemb, code_features], dim=-1)
            gate = torch.sigmoid(self.gate_layer(combined))
            
            # 融合：原特征 + (门控 * 代码特征)
            # 这里的 gate 实现了“逐元素”的筛选
            xemb = xemb + gate * code_features

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y