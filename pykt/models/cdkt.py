import numpy as np
import torch
from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import torch.nn.functional as F

# 直接复用 cakt 中已实现的语义记忆库和因果交叉注意力，避免重复代码
from .cakt import SemanticEncoder, CausalCrossAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CDKT(Module):
    """
    Cross-Attention DKT (CDKT)
    ──────────────────────────────────────────────────────────────
    以标准 DKT（LSTM）为行为主干，通过因果交叉注意力融合 LLM 语义特征，
    增强对学生编程知识状态的建模能力。

    架构概览：
        Branch 1 │ DKT (LSTM) 行为编码器  → H  ∈ R^{B×T×hidden_size}
        Branch 2 │ 纯注意力语义记忆库     → X^K ∈ R^{B×T×hidden_size}
        Fusion   │ 因果 Multi-Head Cross-Attention
                 │ F = LayerNorm(H + Dropout(Context))
        Predict  │ F → Linear(hidden_size, num_c) → Sigmoid → ŷ

    训练目标：
        L_Total = L_BCE + λ · L_MSE
        L_MSE: 用 H_t 预测 e_ks_{t+1}（行为隐空间与语义空间对齐）

    与 CAKT 的核心差异：
        · 行为主干从 AKT (Transformer) 换为 DKT (LSTM)
        · 预测头输出维度为 num_c（所有概念），与 DKT 框架兼容
          推理时通过 one-hot 选出目标概念的预测概率

    参数：
        num_c            概念（知识点）总数
        emb_size         交互嵌入维度（同时也是 LSTM hidden_size）
        dropout          Dropout 概率
        emb_type         嵌入类型（默认 "qid"）
        ks_emb_path      LLM 知识状态向量 .npy 文件路径（冻结嵌入表）
        d_llm            LLM 原始特征维度（ks_emb_path 为空时使用）
        num_attn_heads   Cross-Attention 头数（需整除 emb_size）
        d_ff             语义编码器 FFN 中间层维度（默认 emb_size×2）
        lambda_mse       辅助 MSE Loss 权重 λ
        n_semantic_blocks 语义记忆库 Transformer 块数
    """

    def __init__(
        self,
        num_c: int,
        emb_size: int,
        dropout: float = 0.1,
        emb_type: str = "qid",
        emb_path: str = "",
        ks_emb_path: str = "",
        d_llm: int = 768,
        num_attn_heads: int = 4,
        d_ff: int = 0,
        lambda_mse: float = 0.1,
        n_semantic_blocks: int = 1,
        pretrain_dim: int = 768,
    ):
        super().__init__()

        self.model_name = "cdkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.lambda_mse = lambda_mse

        # d_ff 默认为 emb_size×2
        if d_ff == 0:
            d_ff = emb_size * 2

        # ── Branch 1: DKT (LSTM) 行为编码器 ──────────────────────────────
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(num_c * 2, emb_size)

        self.lstm_layer = LSTM(emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)

        # ── Branch 2: 语义记忆库（含冻结 LLM 嵌入表）─────────────────────
        self.ks_emb_path = ks_emb_path
        if ks_emb_path:
            ks_weight = np.load(ks_emb_path)
            ks_weight = torch.from_numpy(ks_weight).float()
            self.d_llm = ks_weight.shape[1]
            self.ks_emb = Embedding.from_pretrained(ks_weight, freeze=True)
        else:
            self.d_llm = d_llm

        self.semantic_encoder = SemanticEncoder(
            d_llm=self.d_llm,
            d_model=self.hidden_size,
            n_heads=num_attn_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_blocks=n_semantic_blocks,
        )

        # ── Fusion: 因果交叉注意力 ────────────────────────────────────────
        self.cross_attn_fusion = CausalCrossAttention(
            self.hidden_size, num_attn_heads, dropout
        )

        # ── 主预测头（DKT 风格：对全部概念输出概率向量）─────────────────
        self.out_layer = Linear(self.hidden_size, num_c)

        # ── 辅助任务投影：H_t → ê_ks_{t+1} ──────────────────────────────
        self.aux_proj = Linear(self.hidden_size, self.d_llm)

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        只对 mask=1 的有效位计算 MSE，排除 padding 噪声梯度。

        Args:
            pred:   (B, T', d_llm)
            target: (B, T', d_llm)
            mask:   (B, T')  1=有效位, 0=padding；为 None 时退化为普通 MSE
        """
        if mask is None:
            return F.mse_loss(pred, target)
        valid_mask = mask.bool().unsqueeze(-1)              # (B, T', 1)
        n_valid = valid_mask.sum() * pred.size(-1)
        if n_valid == 0:
            return torch.tensor(0.0, device=pred.device)
        diff_sq = (pred - target) ** 2 * valid_mask
        return diff_sq.sum() / n_valid

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        """
        前向传播。

        Args:
            q: (B, T)  概念 ID 序列
            r: (B, T)  作答结果 {0, 1} 序列
            s: (B, T)  提交记录 ID，用于检索冻结 LLM 知识状态向量（必须提供）

        Returns:
            y:        (B, T, num_c)  每时刻对全部概念的答对概率（Sigmoid 输出）
            mse_loss: scalar         辅助语义对齐损失 L_MSE

        训练时在 model_forward 中使用示例：
            y, mse_loss = model(c.long(), r.long(), s.long())
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)   # 选目标概念
            total_loss = bce_loss + model.lambda_mse * mse_loss
        """
        if s is None:
            raise ValueError(
                "CDKT.forward() 需要提供语义特征索引 s（提交记录 ID 序列）。"
            )

        # ── Step 1: 交互嵌入 ──────────────────────────────────────────────
        if self.emb_type.startswith("qid"):
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)        # (B, T, emb_size)

        # ── Step 2: Branch 1 — DKT (LSTM) 行为编码 ───────────────────────
        h, _ = self.lstm_layer(xemb)              # (B, T, hidden_size)
        h = self.dropout_layer(h)

        # ── Step 3: Branch 2 — 语义记忆库编码 ────────────────────────────
        e_ks = self.ks_emb(s)                     # (B, T, d_llm), 冻结
        X_K = self.semantic_encoder(e_ks)         # (B, T, hidden_size)

        # ── Step 4: Fusion — 因果交叉注意力 ──────────────────────────────
        # F = LayerNorm(H + Dropout(CrossAttn(Q=H, K/V=X^K)))
        fused = self.cross_attn_fusion(h, X_K)    # (B, T, hidden_size)

        # ── Step 5: 主预测（DKT 风格，输出全概念向量）────────────────────
        y = torch.sigmoid(self.out_layer(fused))  # (B, T, num_c)

        # ── Step 6: 辅助任务 — 下一时刻语义状态预测 L_MSE ───────────────
        # aux_pred[t] 预测 e_ks[t+1]，需要 t 和 t+1 均为有效位
        # masks: (B, T-1)，其中 masks[b,t]=1 iff 原始序列 t 和 t+1 均不是 padding
        # aux_pred/target 形状为 (B, T-2)，对应 masks[:, :-1]
        aux_pred   = self.aux_proj(h[:, :-1, :])   # (B, T-2, d_llm)
        aux_target = e_ks[:, 1:, :].detach()        # (B, T-2, d_llm)
        mse_mask   = masks[:, :-1] if masks is not None else None  # (B, T-2)
        mse_loss   = self._masked_mse(aux_pred, aux_target, mse_mask)
        # mse_loss = F.mse_loss(aux_pred, aux_target)
        return y, mse_loss
