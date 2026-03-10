import os


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, Sequential, ReLU, GRU
# class ProjectionHead(nn.Module):
#     """
#     将 h 映射到对比空间 Z
#     Input:  (B, T, d_model)
#     Output: (B, T, proj_dim)
#     """
#     def __init__(self, input_dim, proj_dim=128):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.ReLU(),
#             nn.Linear(input_dim, proj_dim)
#         )

#     def forward(self, h):
#         # h: (B, T, d_model)
#         z = self.proj(h)
#         z = F.normalize(z, dim=-1)  # 归一化用于 InfoNCE
#         return z
# class GRUBranch(Module):
#     """
#     纯认知状态编码器
#     """
#     def __init__(self, llm_dim=768, d_model=256):
#         super().__init__()

#         self.proj = Linear(llm_dim, d_model)

#         self.gru = GRU(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             batch_first=True
#         )

#     def forward(self, know_emb):
#         """
#         know_emb: (B, T, 768)
#         """
#                 # ================= 时序对齐修复 (Right Shift) =================
#         B, T, D = know_emb.size()
#         # 构造一个全 0 的初始特征，代表时刻 0 (没有任何交互时的先验状态)
#         zero_pad = torch.zeros(B, 1, D, device=know_emb.device, dtype=know_emb.dtype)
#         # 拼接并在末尾截断，使得长度保持 T: 
#         #[0, emb_1, emb_2, ..., emb_{T-1}]
#         shifted_know_emb = torch.cat([zero_pad, know_emb[:, :-1, :]], dim=1)

#         x = self.proj(shifted_know_emb)  # (B, T, d_model)
#         h_t2, _ = self.gru(x)    # (B, T, d_model)
#         return h_t2
# class ContrastiveLoss(nn.Module):
#     """
#     时间对齐版本：
#     对每个时间步 t：
#         h1[:, t] 和 h2[:, t] 为正样本
#         同 batch 内其他样本为负样本
#     """

#     def __init__(self, temperature=0.2):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, z1, z2, mask=None):
#         """
#         z1, z2: (B, T, D)
#         mask: (B, T)
#         """

#         B, T, D = z1.shape

#         # ===== 1️⃣ L2 归一化（必须开）=====
#         z1 = F.normalize(z1, dim=-1)
#         z2 = F.normalize(z2, dim=-1)

#         total_loss = 0
#         valid_steps = 0

#         for t in range(T):

#             z1_t = z1[:, t, :]   # (B, D)
#             z2_t = z2[:, t, :]   # (B, D)

#             if mask is not None:
#                 mask_t = mask[:, t].bool()
#                 if mask_t.sum() < 2:
#                     continue
#                 z1_t = z1_t[mask_t]
#                 z2_t = z2_t[mask_t]

#             # 相似度矩阵 (B, B)
#             logits = torch.matmul(z1_t, z2_t.T) / self.temperature

#             labels = torch.arange(z1_t.size(0), device=z1.device)

#             loss_t = F.cross_entropy(logits, labels)

#             total_loss += loss_t
#             valid_steps += 1

#         if valid_steps == 0:
#             return torch.tensor(0.0, device=z1.device)

#         return total_loss / valid_steps
#     """
#     同时间步 z1 和 z2 为正样本
#     其余为负样本
#     """
#     def __init__(self, temperature=0.2):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, z1, z2, mask=None):
#         """
#         z1, z2: (B, T, proj_dim)
#         """

#         B, T, D = z1.shape
#         # =================[必须修改点 1: L2 归一化] =================
#         # 沿着最后一个维度 (proj_dim) 将向量长度归一化为 1
#         # 只有这样，两个向量的点乘结果才是严格的余弦相似度 [-1, 1]
#         # z1 = F.normalize(z1, p=2, dim=-1)
#         # z2 = F.normalize(z2, p=2, dim=-1)

#         z1 = z1.reshape(B*T, D)
#         z2 = z2.reshape(B*T, D)

#         if mask is not None:
#             # mask 展平: (B, T) -> (B * T,)
#             mask_flat = mask.reshape(-1).bool()
            
#             # 仅保留真实的做题记录，抛弃 padding 的全0位置
#             z1 = z1[mask_flat]  # 形状变为 (N_valid, D)
#             z2 = z2[mask_flat]  # 形状变为 (N_valid, D)

#         logits = torch.matmul(z1, z2.T) / self.temperature  # (BT, BT)

#         labels = torch.arange(z1.size(0), device=z1.device)

#         loss = F.cross_entropy(logits, labels)

#         return loss
# class GatedFusionPredictor(nn.Module):
#     def __init__(self, d_model, num_c):
#         super().__init__()

#         self.gate = nn.Sequential(
#             nn.Linear(d_model * 2, d_model),
#             nn.Sigmoid()
#         )

#         self.predict = nn.Sequential(
#             # nn.Linear(d_model * 2, d_model),
#             # nn.ReLU(),
#             nn.Linear(d_model, num_c)
#         )

#     def forward(self, h1, h2, e_q_next=None):
#         """
#         h1: (B, T, d_model)
#         h2: (B, T, d_model)
#         e_q_next: (B, T, d_model)
#         """

#         # concat = torch.cat([h1, h2], dim=-1)
#         y = torch.sigmoid(self.predict(0.4*h1+0.6*h2)).squeeze(-1)

#         # g = self.gate(concat)  # (B, T, d_model)

#         # h_fuse = g * h1 + (1 - g) * h2

#         # # # 与下一题 embedding 结合
#         # # pred_input = torch.cat([h_fuse, e_q_next], dim=-1)
#         # pred_input = h_fuse

#         # y = torch.sigmoid(self.predict(pred_input)).squeeze(-1)

#         #(B,T,num_C)
#         return y

# class DKT(Module):
#     def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', ta_emb_path="", ks_emb_path="", pretrain_dim=768):
#         super().__init__()
#         self.model_name = "dkt"
#         self.num_c = num_c
#         self.emb_size = emb_size
#         self.hidden_size = emb_size
#         self.emb_type = emb_type

#         self.ta_emb_path = ta_emb_path
#         self.ks_emb_path = ks_emb_path
#         self.proj_dim=128


#         if emb_type.startswith("qid"):
#             self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

#         lstm_input_dim = self.emb_size

#         if self.ta_emb_path:
#             ta_weight = np.load(self.ta_emb_path)
#             ta_weight = torch.from_numpy(ta_weight).float()
#             self.ta_dim = ta_weight.shape[1]
#             self.ta_emb = Embedding.from_pretrained(
#                 ta_weight, freeze=True
#             )
#             self.fusion = torch.nn.Sequential(
#                 Linear(self.emb_size+self.ta_dim, self.emb_size),
#                 torch.nn.ReLU(),
#             )

#         if self.ks_emb_path:
#             ks_weight = np.load(self.ks_emb_path)
#             ks_weight = torch.from_numpy(ks_weight).float()
#             self.ks_dim = ks_weight.shape[1]
#             self.ks_emb = Embedding.from_pretrained(
#                 ks_weight, freeze=True
#             )

#         self.lstm_layer = LSTM(lstm_input_dim, self.hidden_size, batch_first=True)
#         #=====================================================
#         self.lstm_layer2 = LSTM(lstm_input_dim, self.hidden_size, batch_first=True)
#         self.fusion2 = torch.nn.Sequential(
#             Linear(self.emb_size+self.ta_dim, self.emb_size),
#             torch.nn.ReLU(),
#         )
#         #=====================================================
#         self.gru = GRUBranch(llm_dim=768, d_model=self.hidden_size)
#         self.proj1 = ProjectionHead(self.hidden_size, self.proj_dim)
#         self.proj2 = ProjectionHead(self.hidden_size, self.proj_dim)

#         self.dropout_layer = Dropout(dropout)

#         self.cl_loss = ContrastiveLoss(temperature=0.2)

#         self.predictor = GatedFusionPredictor(self.hidden_size, self.num_c)
#         self.cl_weight = 0.5
        

#     def forward(self, q, r, s=None):
#         # print(f"q.shape is {q.shape}")
#         emb_type = self.emb_type
#         if emb_type == "qid":
#             x = q + self.num_c * r
#             xemb = self.interaction_emb(x)
#         # print(f"xemb.shape is {xemb.shape}")
#             if self.ta_emb_path:
#                 if s is None:
#                     raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
#                 ta_emb = self.ta_emb(s)
#                 # xemb = self.fusion(torch.cat([xemb, ta_emb], dim=-1))

#         h1, _ = self.lstm_layer(self.fusion(torch.cat([xemb, ta_emb], dim=-1)))
#         h1 = self.dropout_layer(h1)
#         if self.ks_emb_path:
#             if s is None:
#                 raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
#             ks_emb = self.ks_emb(s)

#             h2, _ = self.lstm_layer2(self.fusion2(torch.cat([xemb, ks_emb], dim=-1)))
#             # h2 = self.gru(ks_emb)
#             h2 = self.dropout_layer(h2)

#             z1 = self.proj1(h1)
#             z2 = self.proj2(h2)
#             mask = None
#             loss_cl = self.cl_loss(z1, z2, mask)

#         y_pred = self.predictor(h1,h2)

        

#         return y_pred

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        

    def forward(self, q, r):
        # print(f"q.shape is {q.shape}")
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        # print(f"xemb.shape is {xemb.shape}")
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y