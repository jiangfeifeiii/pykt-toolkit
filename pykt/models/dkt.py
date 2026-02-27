import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout, Sequential, ReLU

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', ta_emb_path="", ks_emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        self.ta_emb_path = ta_emb_path
        self.ks_emb_path = ks_emb_path

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        lstm_input_dim = self.emb_size
        if self.ta_emb_path:
            ta_weight = np.load(self.ta_emb_path)
            ta_weight = torch.from_numpy(ta_weight).float()
            self.ta_dim = ta_weight.shape[1]

            self.ta_emb = Embedding.from_pretrained(
                ta_weight, freeze=True
            )
            self.x_fusion_mlp = torch.nn.Sequential(
                Linear(self.emb_size+self.ta_dim, self.emb_size),
                torch.nn.ReLU(),
            )


        if self.ks_emb_path:
            ks_weight = np.load(self.ks_emb_path)
            ks_weight = torch.from_numpy(ks_weight).float()
            self.ks_dim = ks_weight.shape[1]

            self.ks_emb = Embedding.from_pretrained(
                ks_weight, freeze=True
            )
            self.h_fusion_mlp = torch.nn.Sequential(
                Linear(self.hidden_size+self.ks_dim, self.hidden_size),
                torch.nn.ReLU(),
            )

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
            if self.ta_emb_path:
                if s is None:
                    raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
                ta_emb = self.ta_emb(s)
                combined = torch.cat([xemb, ta_emb], dim=-1)
                # xemb = xemb + self.x_fusion_mlp(combined)
                xemb = self.x_fusion_mlp(combined)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        if self.ks_emb_path:
            if s is None:
                raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
            ks_emb = self.ks_emb(s)
            combined = torch.cat([h, ks_emb], dim=-1)
            # h = h+ self.h_fusion_mlp(combined)
            h = self.h_fusion_mlp(combined)
        
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y