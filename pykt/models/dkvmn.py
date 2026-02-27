import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

class DKVMN(Module):
    def __init__(self, num_c, dim_s, size_m, dropout=0.2, emb_type='qid',ta_emb_path="", ks_emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type
        
        self.ta_emb_path = ta_emb_path
        self.ks_emb_path = ks_emb_path

        self.layer_norm = torch.nn.LayerNorm(self.dim_s)
        
        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c, self.dim_s)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))
        if self.ta_emb_path:
            ta_weight = np.load(self.ta_emb_path)
            ta_weight = torch.from_numpy(ta_weight).float()
            self.ta_dim = ta_weight.shape[1]

            self.ta_emb = Embedding.from_pretrained(
                ta_weight, freeze=True
            )
            self.x_fusion_mlp = torch.nn.Sequential(
                Linear(self.dim_s+self.ta_dim, self.dim_s),
                torch.nn.ReLU(),
            )


        if self.ks_emb_path:
            ks_weight = np.load(self.ks_emb_path)
            ks_weight = torch.from_numpy(ks_weight).float()
            self.ks_dim = ks_weight.shape[1]

            self.ks_emb = Embedding.from_pretrained(
                ks_weight, freeze=True
            )
            self.memory_fusion_mlp = torch.nn.Sequential(
                Linear(self.dim_s+self.ks_dim, self.dim_s),
                torch.nn.Tanh(),
            )

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_c * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, q, r, s=None, qtest=False):
        emb_type = self.emb_type
        batch_size = q.shape[0]
        if emb_type == "qid":
            x = q + self.num_c * r
            k = self.k_emb_layer(q)
            v = self.v_emb_layer(x)

            if self.ta_emb_path:
                if s is None:
                    raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
                ta_emb = self.ta_emb(s)
                combined = torch.cat([v, ta_emb], dim=-1)
                v = v + self.x_fusion_mlp(combined)
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        if self.ks_emb_path:
            if s is None:
                raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
            ks_emb = self.ks_emb(s)
            for et,at,wt,ks_t in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2), ks_emb.permute(1, 0, 2)
            ):
                Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                        (wt.unsqueeze(-1) * at.unsqueeze(1))
                combined = torch.cat([Mvt, ks_t.unsqueeze(1).expand(-1, self.size_m, -1)], dim=-1)
                M_correction = self.memory_fusion_mlp(combined)
                Mvt = Mvt+M_correction
                Mvt = self.layer_norm(Mvt)
                Mv.append(Mvt)
        else:
            for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
            ):
                Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                    (wt.unsqueeze(-1) * at.unsqueeze(1))
                Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = self.p_layer(self.dropout_layer(f))

        p = torch.sigmoid(p)
        # print(f"p: {p.shape}")
        p = p.squeeze(-1)
        if not qtest:
            return p
        else:
            return p, f