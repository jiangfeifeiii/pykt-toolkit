import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .akt import Architecture, CosinePositionalEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Branch 2 — Semantic Memory Bank
# ══════════════════════════════════════════════════════════════════════════════

class SemanticTransformerBlock(nn.Module):
    """
    语义记忆库的基础 Transformer 块。
    使用因果自注意力（Causal Self-Attention）保证 t 时刻只能看到 ≤t 的语义历史，
    后接 Position-wise FFN + 残差 & LayerNorm（Pre-LN 风格）。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x: (B, T, d_model)
        """
        T = x.size(1)
        # 上三角置 True → 屏蔽未来位置（PyTorch MHA 中 True 表示忽略该位置）
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        # Pre-LN: 先 Norm 再 Attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.drop1(attn_out)

        normed = self.norm2(x)
        x = x + self.drop2(self.ff(normed))
        return x


class SemanticEncoder(nn.Module):
    """
    Branch 2 — 语义记忆库 (Semantic Memory Bank)

    将冻结的 LLM 特征序列 E_ks ∈ R^{B×T×d_llm} 经过：
        1. 线性投影对齐模型维度
        2. 正弦余弦位置编码 (Sinusoidal PE)
        3. n_blocks 层因果 Transformer 块

    输出语义特征表示 X^K ∈ R^{B×T×d_model}，作为交叉注意力的 Key/Value 来源。
    """

    def __init__(
        self,
        d_llm: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        n_blocks: int = 1,
    ):
        super().__init__()
        self.proj = nn.Linear(d_llm, d_model)
        self.pos_emb = CosinePositionalEmbedding(d_model)
        self.layers = nn.ModuleList(
            [SemanticTransformerBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, e_ks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_ks: (B, T, d_llm)  冻结 LLM 语义向量序列
        Returns:
            X_K:  (B, T, d_model)
        """
        x = self.proj(e_ks)           # (B, T, d_model)
        x = x + self.pos_emb(x)       # 叠加位置编码
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(x)        # 最终 LayerNorm


# ══════════════════════════════════════════════════════════════════════════════
# Fusion — Causal Multi-Head Cross-Attention
# ══════════════════════════════════════════════════════════════════════════════

class CausalCrossAttention(nn.Module):
    """
    因果交叉注意力融合模块 (Causal Cross-Attention Fusion)

    · Q  = H^A  （行为编码分支输出，代表学生当前行为状态）
    · K/V = X^K （语义记忆库输出，代表学生历史认知语义档案）

    因果掩码保证：第 t 时刻的 Query 只能 Attend 到 ≤t 时刻的 Key/Value，
    彻底杜绝未来信息泄露（Data Leakage）。

    残差融合：F = LayerNorm(H^A + Dropout(Context))
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_A: torch.Tensor, X_K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_A: (B, T, d_model)  行为分支隐状态序列
            X_K: (B, T, d_model)  语义记忆库特征序列
        Returns:
            fused: (B, T, d_model)  融合后特征 F
        """
        T = H_A.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=H_A.device, dtype=torch.bool), diagonal=1
        )
        context, _ = self.cross_attn(H_A, X_K, X_K, attn_mask=causal_mask)
        fused = self.norm(H_A + self.dropout(context))
        return fused


# ══════════════════════════════════════════════════════════════════════════════
# Main Model — CAKT
# ══════════════════════════════════════════════════════════════════════════════

class CAKT(nn.Module):
    """
    Cross-Attention Knowledge Tracing with LLM Semantics (CAKT)
    ─────────────────────────────────────────────────────────────
    双分支交叉注意力知识追踪模型，面向编程教育场景。

    架构概览：
        Branch 1 │ AKT 行为编码器          → H^A ∈ R^{B×T×d_model}
        Branch 2 │ 纯注意力语义记忆库      → X^K ∈ R^{B×T×d_model}
        Fusion   │ 因果 Multi-Head Cross-Attention
                 │ F = LayerNorm(H^A + Context)
        Predict  │ F ⊕ q_embed → MLP → Sigmoid → ŷ

    训练目标：
        L_Total = L_BCE + λ · L_MSE
        其中 L_MSE 为辅助任务损失：用 H^A_t 预测 e_ks_{t+1}，
        迫使行为隐空间与 LLM 语义空间对齐。

    参数：
        n_question      题目总数
        n_pid           题目实例总数（用于 Rasch 难度建模，0 表示不启用）
        d_model         模型隐层维度
        n_blocks        AKT 行为编码器 Transformer 块数
        dropout         Dropout 概率
        d_ff            FFN 中间层维度
        kq_same         AKT 中 Key/Query 是否共享权重（1=共享）
        final_fc_dim    预测头第一层维度
        num_attn_heads  Multi-Head Attention 头数
        separate_qa     作答向量是否与题目向量分开嵌入
        l2              Rasch 难度参数 L2 正则系数
        emb_type        嵌入类型（默认 "qid"）
        ks_emb_path     LLM 语义向量 .npy 文件路径（冻结嵌入表）
        d_llm           LLM 原始特征维度（ks_emb_path 为空时使用）
        lambda_mse      辅助 MSE Loss 权重 λ
        n_semantic_blocks  语义记忆库 Transformer 块数
    """

    def __init__(
        self,
        n_question: int,
        n_pid: int,
        d_model: int,
        n_blocks: int,
        dropout: float,
        d_ff: int = 256,
        kq_same: int = 1,
        final_fc_dim: int = 512,
        num_attn_heads: int = 8,
        separate_qa: bool = False,
        l2: float = 1e-5,
        emb_type: str = "qid",
        emb_path: str = "",
        ks_emb_path: str = "",
        d_llm: int = 768,
        lambda_mse: float = 0.1,
        n_semantic_blocks: int = 1,
        pretrain_dim: int = 768,
    ):
        super().__init__()

        self.model_name = "cakt"
        self.n_question = n_question
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = "akt"       # 复用 AKT 的 Architecture
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.lambda_mse = lambda_mse

        embed_l = d_model

        # ── Rasch 难度参数（可选）─────────────────────────────────────────
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        # ── 题目 & 作答嵌入层 ─────────────────────────────────────────────
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        # ── Branch 1: AKT 行为编码器 ──────────────────────────────────────
        self.behavioral_encoder = Architecture(
            n_question=n_question,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model // num_attn_heads,
            d_ff=d_ff,
            kq_same=kq_same,
            model_type=self.model_type,
            emb_type=emb_type,
        )

        # ── Branch 2: 语义记忆库（含冻结 LLM 嵌入表）─────────────────────
        self.ks_emb_path = ks_emb_path
        if self.ks_emb_path:
            ks_weight = np.load(self.ks_emb_path)
            ks_weight = torch.from_numpy(ks_weight).float()
            self.d_llm = ks_weight.shape[1]
            self.ks_emb = nn.Embedding.from_pretrained(ks_weight, freeze=True)
        else:
            self.d_llm = d_llm

        self.semantic_encoder = SemanticEncoder(
            d_llm=self.d_llm,
            d_model=d_model,
            n_heads=num_attn_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_blocks=n_semantic_blocks,
        )

        # ── Fusion: 因果交叉注意力 ────────────────────────────────────────
        self.cross_attn_fusion = CausalCrossAttention(d_model, num_attn_heads, dropout)

        # ── 主预测头：F ⊕ q_embed → ŷ ─────────────────────────────────────
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # ── 辅助任务投影层：H^A_t → ê_ks_{t+1} ──────────────────────────
        self.aux_proj = nn.Linear(d_model, self.d_llm)

        self._reset_params()

    def _reset_params(self):
        """将 Rasch 难度嵌入初始化为 0（遵循原始 AKT 设定）。"""
        for p in self.parameters():
            if self.n_pid > 0 and p.size(0) == self.n_pid + 1:
                torch.nn.init.constant_(p, 0.0)

    # ── 内部辅助方法 ──────────────────────────────────────────────────────────

    def _base_emb(self, q_data: torch.Tensor, target: torch.Tensor):
        """构造题目嵌入 q_embed 和题目-作答联合嵌入 qa_embed。"""
        q_embed = self.q_embed(q_data)               # (B, T, d_model)
        if self.separate_qa:
            qa_idx = q_data + self.n_question * target
            qa_embed = self.qa_embed(qa_idx)
        else:
            qa_embed = self.qa_embed(target) + q_embed   # c_ct + g_rt
        return q_embed, qa_embed

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        q_data: torch.Tensor,
        target: torch.Tensor,
        s: torch.Tensor = None,
        pid_data: torch.Tensor = None,
        qtest: bool = False,
    ):
        """
        前向传播。

        Args:
            q_data:   (B, T)  题目 ID 序列
            target:   (B, T)  作答结果 {0, 1} 序列
            s:        (B, T)  提交记录 ID，用于检索冻结 LLM 语义向量（必须提供）
            pid_data: (B, T)  题目实例 ID（启用 Rasch 难度时必须提供）
            qtest:    bool    若为 True，额外返回中间融合表示（可解释性分析用）

        Returns (qtest=False):
            preds       (B, T)  每时刻答对概率 ŷ（Sigmoid 输出）
            c_reg_loss  scalar  Rasch L2 正则项（n_pid=0 时恒为 0）
            mse_loss    scalar  辅助语义对齐损失 L_MSE

        训练时总损失计算示例：
            bce_loss = F.binary_cross_entropy(preds[:, :-1], target[:, 1:].float(), ...)
            total_loss = bce_loss + c_reg_loss + model.lambda_mse * mse_loss
        """
        if s is None:
            raise ValueError(
                "CAKT.forward() 需要提供语义特征索引 s（提交记录 ID 序列）。"
            )

        # ── Step 1: Embedding ─────────────────────────────────────────────
        q_embed, qa_embed = self._base_emb(q_data, target)

        pid_embed = None
        if self.n_pid > 0:
            if pid_data is None:
                raise ValueError("n_pid > 0 时需提供 pid_data。")
            q_embed_diff = self.q_embed_diff(q_data)         # d_ct
            pid_embed = self.difficult_param(pid_data)        # u_q, (B,T,1)
            q_embed = q_embed + pid_embed * q_embed_diff      # u_q·d_ct + c_ct

            qa_embed_diff = self.qa_embed_diff(target)        # f_(ct,rt)
            if self.separate_qa:
                qa_embed = qa_embed + pid_embed * qa_embed_diff
            else:
                qa_embed = qa_embed + pid_embed * (qa_embed_diff + q_embed_diff)

            c_reg_loss = (pid_embed ** 2.0).sum() * self.l2
        else:
            c_reg_loss = 0.0

        # ── Step 2: Branch 1 — 行为编码 (AKT Backbone) ───────────────────
        # H_A: (B, T, d_model)
        H_A = self.behavioral_encoder(q_embed, qa_embed, pid_embed)

        # ── Step 3: Branch 2 — 语义记忆库编码 ────────────────────────────
        e_ks = self.ks_emb(s)                  # (B, T, d_llm), 冻结
        X_K = self.semantic_encoder(e_ks)      # (B, T, d_model)

        # ── Step 4: Fusion — 因果交叉注意力 ──────────────────────────────
        # F = LayerNorm(H^A + Dropout(CrossAttn(Q=H^A, K/V=X^K)))
        fused = self.cross_attn_fusion(H_A, X_K)   # (B, T, d_model)

        # ── Step 5: 主预测任务 ────────────────────────────────────────────
        # concat fused state with question embedding, then predict
        concat_q = torch.cat([fused, q_embed], dim=-1)   # (B, T, 2·d_model)
        logits = self.out(concat_q).squeeze(-1)           # (B, T)
        preds = torch.sigmoid(logits)

        # ── Step 6: 辅助任务 — 下一语义状态预测 L_MSE ────────────────────
        # 用 H^A_t 预测 e_ks_{t+1}，对齐行为隐空间与 LLM 语义空间
        aux_pred = self.aux_proj(H_A[:, :-1, :])          # (B, T-1, d_llm)
        aux_target = e_ks[:, 1:, :].detach()              # (B, T-1, d_llm)
        mse_loss = F.mse_loss(aux_pred, aux_target)

        if qtest:
            return preds, c_reg_loss, mse_loss, concat_q
        return preds, c_reg_loss, mse_loss
