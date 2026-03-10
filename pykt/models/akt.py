import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
print("cross_attn")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
class ProjectionHead(nn.Module):
    """
    将 h 映射到对比空间 Z
    Input:  (B, T, d_model)
    Output: (B, T, proj_dim)
    """
    def __init__(self, input_dim, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim)
        )

    def forward(self, h):
        # h: (B, T, d_model)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)  # 归一化用于 InfoNCE
        return z
class AttnBranch(nn.Module):
    """
    微观行为编码器
    """
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type
        self.tech_proj = nn.Linear(768, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU()
        )
        self.attn = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,  kq_same=kq_same, model_type=model_type, emb_type=emb_type)

    def forward(self, q_embed_data, r_embed_data, ta_embed_data, pid_embed_data=None):
        """
        know_emb: (B, T, 768)
        """
        ta_embed_data = self.tech_proj(ta_embed_data)
        qa_embed_data = self.fusion(torch.cat([q_embed_data, r_embed_data, ta_embed_data], dim=-1))
        h_t1 = self.attn(q_embed_data, qa_embed_data, pid_embed_data)    # (B, T, d_model)
        return h_t1
class GRUBranch(nn.Module):
    """
    纯认知状态编码器
    """
    def __init__(self, llm_dim=768, d_model=256, num_layers= 1):
        super().__init__()

        self.proj = nn.Linear(llm_dim, d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout = 0.2,
        )

    def forward(self, know_emb):
        """
        know_emb: (B, T, 768)
        """
                # ================= 时序对齐修复 (Right Shift) =================
        B, T, D = know_emb.size()
        # 构造一个全 0 的初始特征，代表时刻 0 (没有任何交互时的先验状态)
        zero_pad = torch.zeros(B, 1, D, device=know_emb.device, dtype=know_emb.dtype)
        # 拼接并在末尾截断，使得长度保持 T: 
        #[0, emb_1, emb_2, ..., emb_{T-1}]
        shifted_know_emb = torch.cat([zero_pad, know_emb[:, :-1, :]], dim=1)

        x = self.proj(shifted_know_emb)  # (B, T, d_model)
        h_t2, _ = self.gru(x)    # (B, T, d_model)
        return h_t2
class ContrastiveLoss(nn.Module):
    """
    同时间步 z1 和 z2 为正样本
    其余为负样本
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, mask=None):
        """
        z1, z2: (B, T, proj_dim)
        """

        B, T, D = z1.shape
        # =================[必须修改点 1: L2 归一化] =================
        # 沿着最后一个维度 (proj_dim) 将向量长度归一化为 1
        # 只有这样，两个向量的点乘结果才是严格的余弦相似度 [-1, 1]
        # z1 = F.normalize(z1, p=2, dim=-1)
        # z2 = F.normalize(z2, p=2, dim=-1)

        z1 = z1.reshape(B*T, D)
        z2 = z2.reshape(B*T, D)

        if mask is not None:
            # mask 展平: (B, T) -> (B * T,)
            mask_flat = mask.reshape(-1).bool()
            
            # 仅保留真实的做题记录，抛弃 padding 的全0位置
            z1 = z1[mask_flat]  # 形状变为 (N_valid, D)
            z2 = z2[mask_flat]  # 形状变为 (N_valid, D)

        logits = torch.matmul(z1, z2.T) / self.temperature  # (BT, BT)

        labels = torch.arange(z1.size(0), device=z1.device)

        loss = F.cross_entropy(logits, labels)

        return loss
class GatedFusionPredictor(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.predict = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, h1, h2, e_q_next):
        """
        h1: (B, T, d_model)
        h2: (B, T, d_model)
        e_q_next: (B, T, d_model)
        """

        concat = torch.cat([h1, h2], dim=-1)

        g = self.gate(concat)  # (B, T, d_model)

        h_fuse = g * h1 + (1 - g) * h2

        # 与下一题 embedding 结合
        pred_input = torch.cat([h_fuse, e_q_next], dim=-1)

        y = torch.sigmoid(self.predict(pred_input)).squeeze(-1)

        return y

# class DecoupledLLMContrastiveKT(nn.Module):
# class AKT(nn.Module):
#     def __init__(self,
#                  n_question,
#                  n_pid,
#                  d_model,
#                  n_blocks,
#                  dropout,
#                  d_ff=256,
#                  kq_same=1,
#                  final_fc_dim=512,
#                  num_attn_heads=8,
#                  separate_qa=False,
#                  l2=1e-5,
#                  emb_type="qid",
#                  ta_emb_path="",
#                  ks_emb_path="",
#                  pretrain_dim=768,
#                  llm_dim=768,
#                  proj_dim=128):

#         super().__init__()
#         self.model_name = "akt"
#         self.n_question = n_question
#         self.dropout = dropout
#         self.kq_same = kq_same
#         self.n_pid = n_pid
#         self.l2 = l2
#         self.model_type = self.model_name
#         self.separate_qa = separate_qa
#         self.emb_type = emb_type

#         self.cl_weight = 0.5

#         self.ta_emb_path = ta_emb_path
#         self.ks_emb_path = ks_emb_path

#         if emb_type.startswith("qid"):
#             # n_question+1 ,d_model
#             self.q_embed = nn.Embedding(self.n_question, d_model)
#             if self.separate_qa: 
#                 self.qa_embed = nn.Embedding(2*self.n_question+1, d_model) # interaction emb
#             else: # false default
#                 self.qa_embed = nn.Embedding(2, d_model)
#         if self.ta_emb_path:
#             ta_weight = np.load(self.ta_emb_path)
#             ta_weight = torch.from_numpy(ta_weight).float()
#             self.ta_dim = ta_weight.shape[1]
#             self.ta_emb = nn.Embedding.from_pretrained(
#                 ta_weight, freeze=True
#             )    
#         if self.ks_emb_path:
#             ks_weight = np.load(self.ks_emb_path)
#             ks_weight = torch.from_numpy(ks_weight).float()
#             self.ks_dim = ks_weight.shape[1]
#             self.ks_emb = nn.Embedding.from_pretrained(
#                 ks_weight, freeze=True
#             )
#         self.norm = nn.LayerNorm(d_model)


#         self.attn = AttnBranch(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
#                                     d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)
#         self.gru = GRUBranch(llm_dim=768, d_model=d_model)

#         self.proj1 = ProjectionHead(d_model, proj_dim)
#         self.proj2 = ProjectionHead(d_model, proj_dim)

#         self.cl_loss = ContrastiveLoss(temperature=0.2)

#         self.predictor = GatedFusionPredictor(d_model)
#     def forward(self, q, r, s=None, pid_data=None, qtest=False):
#         """
#         q: (B, T)
#         r: (B, T)
#         tech_emb: (B, T, 768)
#         know_emb: (B, T, 768)
#         """
#         emb_type = self.emb_type
#         # Batch First
#         if emb_type.startswith("qid"):
#             q_embed_data = self.q_embed(q)
#             r_embed_data = self.qa_embed(r)
#             ta_embed_data = self.ta_emb(s)
#             ks_embed_data = self.ks_emb(s)
#         # ====== Branch 1 ======
#         h1 = self.attn(q_embed_data, r_embed_data, ta_embed_data)  # (B, T, d)
#         # ====== Branch 2 ======
#         h2 = self.gru(ks_embed_data)             # (B, T, d)

#         # ====== Contrastive ======
#         z1 = self.proj1(h1)
#         z2 = self.proj2(h2)
#         mask = None
#         # mask = (q != -1)       
#         # for i in range(q.size(0)):
#         #     if (q[i] == -1).any():
#         #         print("q:")
#         #         print(q[i])
#         #         print("mask:")
#         #         print(mask[i])
#         #         print("="*50)
#         loss_cl = self.cl_loss(z1, z2, mask)

#         # ====== Next Question Prediction ======
#         # e_q_next = q_embed_data[:, 1:, :]
#         # h1 = h1[:, :-1, :]
#         # h2 = h2[:, :-1, :]

#         y_pred = self.predictor(h1, h2, q_embed_data)

#         rasch_reg = 0
#         total_loss = rasch_reg + self.cl_weight*loss_cl
#         return y_pred, total_loss
class GRUMemoryRetriever(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h_akt, h_gru, mask=None):
        """
        h_akt: (B, T, d_model)  —— 当前时刻的“认知需求”表示，来自 AKT 分支
        h_gru: (B, T, d_model)  —— 纯知识记忆轨迹，来自 GRU 分支
        mask:  (B, T) bool      —— padding mask，可选
        """
        B, T, D = h_akt.size()

        q = self.q_linear(h_akt).view(B, T, self.h, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.k_linear(h_gru).view(B, T, self.h, self.d_k).transpose(1, 2) # (B, H, T, d_k)
        v = self.v_linear(h_gru).view(B, T, self.h, self.d_k).transpose(1, 2) # (B, H, T, d_k)

        # 构造“只能看历史”的因果 mask：每个 t 只能看到 <= t 的 h_gru
        causal = torch.tril(torch.ones(T, T, device=h_akt.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        # scores: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(causal == 0, float('-inf'))

        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T)
            pad_mask = (~mask).unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(pad_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # (B, H, T, d_k)
        context = torch.matmul(attn, v)

        # (B, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(context)  # (B, T, D)

        return out
class AKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout, d_ff=256, 
            kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", ta_emb_path = "", ks_emb_path="", pretrain_dim=768, num_layers):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "akt"
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1) # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb, 同上
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l) # interaction emb
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, emb_type=self.emb_type)
        self.ta_emb_path = ta_emb_path
        self.ks_emb_path = ks_emb_path
        if self.ks_emb_path:
            self.out = nn.Sequential(
                nn.Linear(2*d_model + embed_l,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(d_model + embed_l,
                        final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )
        self.reset()

        if self.ta_emb_path:
            print(self.ta_emb_path)
            ta_weight = np.load(self.ta_emb_path)
            ta_weight = torch.from_numpy(ta_weight).float()
            self.ta_dim = ta_weight.shape[1]
            self.ta_emb = nn.Embedding.from_pretrained(
                ta_weight, freeze=True
            )
            self.x_fusion_mlp = torch.nn.Sequential(
                nn.Linear(embed_l+self.ta_dim, embed_l),
                torch.nn.ReLU(),
            )
        if self.ks_emb_path:
            ks_weight = np.load(self.ks_emb_path)
            ks_weight = torch.from_numpy(ks_weight).float()
            self.ks_dim = ks_weight.shape[1]
            self.ks_emb = nn.Embedding.from_pretrained(
                ks_weight, freeze=True
            )    
            self.gru = GRUBranch(llm_dim=self.ks_dim, d_model=d_model, num_layers = num_layers)
            self.gru_retriever = GRUMemoryRetriever(d_model=d_model, n_heads=num_attn_heads, dropout=dropout)
    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, q_data, target, s=None, pid_data=None, attn_m = None, qtest=False):
        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2 # rasch部分loss
        else:
            c_reg_loss = 0.

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        if self.ta_emb_path:
            if s is None:
                raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
            ta_emb = self.ta_emb(s)

            combined = torch.cat([qa_embed_data, ta_emb], dim=-1)
            
            qa_embed_data = qa_embed_data + self.x_fusion_mlp(combined)
        if self.ks_emb_path:
            if s is None:
                raise ValueError("模型初始化了 emb_path,但在 forward 时未提供 sub_id")
            ks_emb = self.ks_emb(s)
            h_gru  = self.gru(ks_emb)
            self.proj1 = ProjectionHead(d_model, proj_dim)
            self.proj2 = ProjectionHead(d_model, proj_dim)

            self.cl_loss = ContrastiveLoss(temperature=0.2)
            self.cl_weight = 0.5
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        loss_cl = 0
        if self.ks_emb_path:
            # h_retr = self.gru_retriever(h_akt=d_output, h_gru=h_gru, mask=attn_m)
            # concat_q = torch.cat([d_output, h_retr, q_embed_data], dim=-1)

            #对比学习
            z1 = self.proj1(h1)
            z2 = self.proj2(h2)
            loss_cl = self.cl_loss(z1, z2, attn_m)

            concat_q = torch.cat([d_output, h_gru, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
        else:
            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        if not qtest:
            return preds, c_reg_loss+self.cl_weight*loss_cl
        else:
            return preds, c_reg_loss, concat_q

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, emb_type):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, emb_type=emb_type)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data) # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data) # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, emb_type):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.emb_type = emb_type
        if emb_type.endswith("avgpool"):
            # pooling
            #self.pool =  nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            pool_size = 3
            self.pooling =  nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False, )
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.endswith("linear"):
            # linear
            self.linear = nn.Linear(d_model, d_model, bias=bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        elif emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            torch.nn.init.xavier_uniform_(self.gammas)
            self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            # constant_(self.attnlinear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):

        bs = q.size(0)

        if self.emb_type.endswith("avgpool"):
            # v = v.transpose(1,2)
            scores = self.pooling(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)#.contiguous().view(bs, -1, self.d_model)
        elif self.emb_type.endswith("linear"):
            # v = v.transpose(1,2)
            scores = self.linear(v)
            concat = self.pad_zero(scores, bs, scores.shape[2], zero_pad)
            # concat = concat.transpose(1,2)
        elif self.emb_type.startswith("qid"):
            # perform linear operation and split into h heads

            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * h * sl * d_model

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # calculate attention using function we will define next
            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            scores = attention(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, gammas, pdiff)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1) # 所有v后置一位
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device) # 结果和上一步一样
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
        # print(f"distotal_scores: {disttotal_scores}")
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.) # score <0 时，设置为0
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    if pdiff == None:
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores*gamma*diff).exp(), min=1e-5), max=1e5) # 对应论文公式1中的新增部分
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
