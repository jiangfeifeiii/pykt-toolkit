import numpy as np
import torch
from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CodeDKT(Module):
    """
    Code-DKT: Code-based Deep Knowledge Tracing
    ──────────────────────────────────────────────────────────────
    在标准 DKT (LSTM) 基础上，融合学生代码提交的预训练语义向量，
    增强对编程知识状态的建模能力。

    架构概览：
        Branch 1 │ 标准交互嵌入 embed(c + num_c * r) → xemb ∈ R^{B×T×emb_size}
        Branch 2 │ 代码嵌入查表 code_emb[s]         → e_c  ∈ R^{B×T×d_code} (冻结)
        Proj     │ Linear(d_code → emb_size)         → e_c_proj
        Fusion   │ concat([xemb, e_c_proj]) → Linear(2*emb_size → emb_size)
                 │ 或 "add": xemb + e_c_proj
                 │ 或 "gate": 门控加权
        Backbone │ LSTM(emb_size → hidden_size)
        Predict  │ Linear(hidden_size → num_c) → Sigmoid → ŷ

    若 code_emb_path 为空，自动退化为标准 DKT（无代码特征）。

    参数：
        num_c          概念（知识点）总数
        emb_size       交互嵌入 / LSTM 隐状态维度
        dropout        Dropout 概率
        emb_type       嵌入类型（默认 "qid"）
        code_emb_path  代码嵌入 .npy 文件路径
                       shape: (num_submissions+1, d_code)
                       索引 0 对应 padding（全零行）
        d_code         代码嵌入维度（仅在 code_emb_path 为空时作为占位，实际不使用）
        fusion         融合方式: "concat" | "add" | "gate"
    """

    def __init__(
        self,
        num_c: int,
        emb_size: int,
        dropout: float = 0.1,
        emb_type: str = "qid",
        emb_path: str = "",
        code_emb_path: str = "",
        d_code: int = 768,
        fusion: str = "concat",
        pretrain_dim: int = 768,
    ):
        super().__init__()

        self.model_name = "codedkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.fusion = fusion
        self.code_emb_path = code_emb_path

        # ── Branch 1: 标准交互嵌入 ─────────────────────────────────────────
        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(num_c * 2, emb_size)

        # ── Branch 2: 代码嵌入（冻结预训练向量）──────────────────────────
        if code_emb_path:
            code_weight = np.load(code_emb_path)
            code_weight = torch.from_numpy(code_weight).float()
            self.d_code = code_weight.shape[1]
            self.code_emb = Embedding.from_pretrained(code_weight, freeze=True)
            # 投影到 emb_size
            self.code_proj = Linear(self.d_code, emb_size)
            # 融合层
            if fusion == "concat":
                self.fusion_proj = Linear(emb_size * 2, emb_size)
        else:
            self.d_code = 0  # 无代码特征，退化为标准 DKT

        # ── Backbone: LSTM ────────────────────────────────────────────────
        self.lstm_layer = LSTM(emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, num_c)

    def forward(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor = None,
    ):
        """
        前向传播。

        Args:
            q: (B, T)  概念 ID 序列
            r: (B, T)  作答结果 {0, 1} 序列
            s: (B, T)  提交记录 ID，用于检索代码嵌入（可选）

        Returns:
            y: (B, T, num_c)  每时刻对全部概念的答对概率（Sigmoid 输出）

        使用示例（在 train_model.py 的 model_forward 中）：
            y = model(c.long(), r.long(), s.long())
            y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        """
        # ── Step 1: 交互嵌入 ──────────────────────────────────────────────
        if self.emb_type.startswith("qid"):
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)   # (B, T, emb_size)

        # ── Step 2: 代码特征融合 ─────────────────────────────────────────
        if self.d_code > 0 and s is not None:
            e_code = self.code_emb(s)                    # (B, T, d_code), 冻结
            e_code_proj = self.code_proj(e_code)         # (B, T, emb_size)

            if self.fusion == "concat":
                lstm_input = self.fusion_proj(
                    torch.cat([xemb, e_code_proj], dim=-1)
                )                                        # (B, T, emb_size)
            elif self.fusion == "add":
                lstm_input = xemb + e_code_proj
            elif self.fusion == "gate":
                g = torch.sigmoid(xemb + e_code_proj)   # (B, T, emb_size)
                lstm_input = g * xemb + (1 - g) * e_code_proj
            else:
                lstm_input = xemb
        else:
            lstm_input = xemb

        # ── Step 3: LSTM 序列建模 ────────────────────────────────────────
        h, _ = self.lstm_layer(lstm_input)   # (B, T, hidden_size)
        h = self.dropout_layer(h)

        # ── Step 4: 预测（DKT 风格，输出全概念向量）─────────────────────
        y = torch.sigmoid(self.out_layer(h)) # (B, T, num_c)
        return y
