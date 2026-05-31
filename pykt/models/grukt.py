import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CodeDenoiseGate(nn.Module):
    """Pro-branch only, state-level denoising gate (Coda-style).

    Controls how much of the raw GRU update is accepted for the pro branch,
    conditioned on the CodeBERT embedding of the current submission and,
    optionally, its signal_type (weak / normal).

    Only applied to the pro (problem) branch.
    Does NOT touch the skill/concept branch or the global branch.

    signal_type values (4-class interface for future extensibility):
        0 = core      (v1: not generated)
        1 = weak      (sim_prev > tau_weak_high)
        2 = normal    (default)
        3 = unwanted  (v1: not generated)

    Gate formula:
        gate = sigmoid(MLP_gate(z_t))   # scalar gate, shape [B, 1]
        h_pro_denoised = h_pro_prev + gate * (h_pro_raw - h_pro_prev)
    """

    def __init__(
        self,
        code_emb_dim: int,
        d: int,
        signal_emb_dim: int = 16,
        use_signal_type: bool = True,
        dropout: float = 0.1,
    ):
        super(CodeDenoiseGate, self).__init__()
        self.use_signal_type = use_signal_type

        # Project CodeBERT embedding down to model dimension d
        self.code_proj = nn.Sequential(
            nn.Linear(code_emb_dim, d),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d),
        )

        # signal_type embedding (4 classes, kept as nn.Embedding so it is trainable)
        self.signal_embedding = nn.Embedding(4, signal_emb_dim)

        gate_input_dim = d + signal_emb_dim if use_signal_type else d

        # MLP that produces a scalar gate value in (0, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, d),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        code_emb_t: torch.Tensor,   # [B, code_emb_dim]
        signal_type_t: torch.Tensor, # [B]  long
        h_pro_prev: torch.Tensor,    # [B, d]  last pro state (before this update)
        h_pro_raw: torch.Tensor,     # [B, d]  raw pro_gru output
    ) -> torch.Tensor:
        """Return denoised pro state: h_prev + gate * (h_raw - h_prev)."""
        z = self.code_proj(code_emb_t)          # [B, d]
        if self.use_signal_type:
            s = self.signal_embedding(signal_type_t)  # [B, signal_emb_dim]
            z = torch.cat([z, s], dim=-1)        # [B, d + signal_emb_dim]
        gate = self.gate_mlp(z)                  # [B, 1]
        return h_pro_prev + gate * (h_pro_raw - h_pro_prev)


class GRUKT(nn.Module):
    """GRU-Core Knowledge Tracing model.

    Three independent GRU-based state streams, all updated by ``nn.GRUCell``:
    - global (domain) state: a single hidden vector that is rolled forward at every step
    - per-question state: hidden vector indexed by question id (looked up via ``last_pro_time``)
    - per-concept state: hidden vector indexed by skill id (looked up via ``last_skill_time``)

    Compared with ReKT, this model removes the time-gap forget gates and uses
    GRU as the unified update core for all three branches. Question difficulty
    is modeled as in ReKT (akt_pro_diff * akt_pro_change on top of q/c embeds).
    No LLM semantic branch is included.
    """

    def __init__(
        self,
        skill_max,
        pro_max,
        d,
        dropout,
        max_seq=200,
        emb_type="qid",
        use_score_gate=False,
        use_score_residual=False,
        score_path="",
        # ---- code denoising gate (pro branch only) ----
        use_code_denoise_gate=False,
        use_signal_type=True,
        code_emb_path="",
        signal_type_path="",
        code_emb_dim=768,
        signal_emb_dim=16,
        freeze_backbone=False,
    ):
        super(GRUKT, self).__init__()

        self.pro_max = pro_max
        self.skill_max = skill_max
        self.model_name = "grukt"
        self.emb_type = emb_type
        self.d = d
        self.max_seq = max_seq
        self.use_score_gate = use_score_gate
        self.use_score_residual = use_score_residual
        self.use_code_denoise_gate = use_code_denoise_gate
        self.use_signal_type = use_signal_type
        self.freeze_backbone = freeze_backbone

        self.pro_embed = nn.Parameter(torch.rand(pro_max, d))
        self.skill_embed = nn.Parameter(torch.rand(skill_max, d))
        self.ans_embed = nn.Parameter(torch.rand(2, d))

        self.akt_pro_diff = nn.Parameter(torch.rand(pro_max, 1))
        self.akt_pro_change = nn.Parameter(torch.rand(skill_max, d))

        self.pro_state_init = nn.Parameter(torch.rand(max_seq - 1, d))
        self.skill_state_init = nn.Parameter(torch.rand(max_seq - 1, d))
        self.global_state_init = nn.Parameter(torch.rand(1, d))

        self.global_gru = nn.GRUCell(d, d)
        self.pro_gru = nn.GRUCell(d, d)
        self.skill_gru = nn.GRUCell(d, d)

        self.out = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d, 1),
        )

        self.dropout = nn.Dropout(p=dropout)

        if self.use_score_gate and score_path:
            score_np = np.load(score_path)
            self.register_buffer("score_table", torch.from_numpy(score_np).float())
        else:
            self.register_buffer("score_table", torch.zeros(1, dtype=torch.float))

        if self.use_score_gate:
            # Exp2: score_residual 使输入从 2 维扩展到 3 维 [cmes, signed_cmes, score_residual]
            score_input_dim = 3 if use_score_residual else 2
            self.score_proj = nn.Sequential(
                nn.Linear(score_input_dim, d),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )
            self.skill_score_update_proj = nn.Linear(2 * d, d)
            self.score_update_gate = nn.Sequential(
                nn.Linear(3 * d, d),
                nn.Sigmoid(),
            )

        # ------------------------------------------------------------------ #
        #  Code denoising gate — pro branch only, offline CodeBERT features  #
        # ------------------------------------------------------------------ #
        # Buffers and gate module are only created when use_code_denoise_gate=True,
        # so old checkpoints (without gate) can be loaded without spurious warnings.
        if self.use_code_denoise_gate:
            if code_emb_path:
                code_np = np.load(code_emb_path)
                self.register_buffer("code_emb_table", torch.from_numpy(code_np).float())
            else:
                # placeholder: zero embedding for all sids
                self.register_buffer(
                    "code_emb_table", torch.zeros(1, code_emb_dim, dtype=torch.float)
                )

            if signal_type_path:
                sig_np = np.load(signal_type_path)
                self.register_buffer("signal_type_table", torch.from_numpy(sig_np).long())
            else:
                # placeholder: numel()==1 triggers fallback to normal(2) in forward
                self.register_buffer(
                    "signal_type_table", torch.zeros(1, dtype=torch.long)
                )

        if self.use_code_denoise_gate:
            self.code_denoise_gate = CodeDenoiseGate(
                code_emb_dim=code_emb_dim,
                d=d,
                signal_emb_dim=signal_emb_dim,
                use_signal_type=use_signal_type,
                dropout=dropout,
            )

        # ------------------------------------------------------------------ #
        #  Ablation: freeze backbone, only train code gate parameters         #
        # ------------------------------------------------------------------ #
        if freeze_backbone and self.use_code_denoise_gate:
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.code_denoise_gate.parameters():
                p.requires_grad_(True)

    def forward(self, dcur, qtest=False, train=False):
        next_problem = dcur["shft_qseqs"].long()
        next_skill = dcur["shft_cseqs"].long()
        next_ans = dcur["shft_rseqs"].long()
        next_submission = dcur["shft_sseqs"].long() if "shft_sseqs" in dcur else None

        device = next_problem.device
        batch = next_problem.shape[0]
        seq = next_problem.shape[-1]

        next_pro_embed = (
            F.embedding(next_problem, self.pro_embed)
            + F.embedding(next_skill, self.skill_embed)
            + F.embedding(next_problem, self.akt_pro_diff)
            * F.embedding(next_skill, self.akt_pro_change)
        )  # [B, T, d]
        next_X = next_pro_embed + F.embedding(next_ans, self.ans_embed)  # [B, T, d]

        if seq <= self.pro_state_init.shape[0]:
            pro_state = self.pro_state_init[:seq].unsqueeze(0).repeat(batch, 1, 1)
            skill_state = self.skill_state_init[:seq].unsqueeze(0).repeat(batch, 1, 1)
        else:
            pad = torch.zeros(seq - self.pro_state_init.shape[0], self.d, device=device)
            pro_state = torch.cat([self.pro_state_init, pad], dim=0).unsqueeze(0).repeat(batch, 1, 1)
            skill_state = torch.cat([self.skill_state_init, pad], dim=0).unsqueeze(0).repeat(batch, 1, 1)

        global_state = self.global_state_init.repeat(batch, 1)  # [B, d]

        last_pro_time = torch.zeros((batch, self.pro_max), dtype=torch.long, device=device)
        last_skill_time = torch.zeros((batch, self.skill_max), dtype=torch.long, device=device)

        batch_index = torch.arange(batch, device=device)

        res_p = []
        concat_q = []

        for t in range(seq):
            now_pro = next_problem[:, t]
            now_skill = next_skill[:, t]
            now_pro_embed = next_pro_embed[:, t]

            last_pt = last_pro_time[batch_index, now_pro]
            last_pro_state = pro_state[batch_index, last_pt]

            last_st = last_skill_time[batch_index, now_skill]
            last_skill_state = skill_state[batch_index, last_st]

            final_state = torch.cat(
                [global_state, last_pro_state, last_skill_state, now_pro_embed], dim=-1
            )
            P = torch.sigmoid(self.out(self.dropout(final_state))).squeeze(-1)

            res_p.append(P)
            concat_q.append(final_state)

            x_t = self.dropout(next_X[:, t])
            new_global = self.global_gru(x_t, global_state)
            new_pro = self.pro_gru(x_t, last_pro_state)

            # ---- code denoising gate: pro branch only, no info leak ----
            # Applied after pro_gru raw update, before writing to pro_state.
            # Uses offline CodeBERT embedding and signal_type (weak/normal)
            # to control how much of the raw state update is accepted.
            # Does NOT affect skill branch, global branch, or prediction head.
            if self.use_code_denoise_gate and next_submission is not None:
                sid_t = next_submission[:, t].clamp(min=0)

                sid_code = sid_t.clamp(max=self.code_emb_table.shape[0] - 1)
                code_emb_t = self.code_emb_table[sid_code]  # [B, code_emb_dim]

                if self.signal_type_table.numel() > 1:
                    sid_sig = sid_t.clamp(max=self.signal_type_table.shape[0] - 1)
                    signal_type_t = self.signal_type_table[sid_sig]   # [B]
                else:
                    # fallback: treat all as normal (2)
                    signal_type_t = torch.full_like(sid_t, fill_value=2)

                new_pro = self.code_denoise_gate(
                    code_emb_t=code_emb_t,
                    signal_type_t=signal_type_t,
                    h_pro_prev=last_pro_state,
                    h_pro_raw=new_pro,
                )

            if self.use_score_gate:
                fallback_score = next_ans[:, t].float() * 4.0 + 1.0  # AC->5, non-AC->1
                if next_submission is not None and self.score_table.numel() > 1:
                    sid_t = next_submission[:, t].clamp(min=0)
                    sid_t = sid_t.clamp(max=self.score_table.shape[0] - 1)
                    llm_score = self.score_table[sid_t]
                    score = torch.where(llm_score > 0, llm_score, fallback_score)
                else:
                    score = fallback_score
                cmes = ((score - 1.0) / 4.0).unsqueeze(-1).clamp(0.0, 1.0)
                signed_cmes = 2.0 * cmes - 1.0
                # Exp2: 将 score_residual = cmes - binary_correctness 拼入特征
                # 刻画 LLM score 相对于二元正确性的偏移（细粒度掌握证据）
                if self.use_score_residual:
                    ans_float = next_ans[:, t].float().unsqueeze(-1)
                    score_residual = cmes - ans_float
                    score_feat = torch.cat([cmes, signed_cmes, score_residual], dim=-1)
                else:
                    score_feat = torch.cat([cmes, signed_cmes], dim=-1)
                score_emb = self.score_proj(score_feat)
                skill_update_input = self.skill_score_update_proj(
                    torch.cat([x_t, score_emb], dim=-1)
                )
                candidate_skill = self.skill_gru(skill_update_input, last_skill_state)
                gate = self.score_update_gate(
                    torch.cat([last_skill_state, candidate_skill, score_emb], dim=-1)
                )
                new_skill = gate * candidate_skill + (1.0 - gate) * last_skill_state
            else:
                new_skill = self.skill_gru(x_t, last_skill_state)

            global_state = new_global
            pro_state[:, t] = new_pro
            skill_state[:, t] = new_skill
            last_pro_time[batch_index, now_pro] = t
            last_skill_time[batch_index, now_skill] = t

        res_p_t = torch.stack(res_p, dim=1)  # [B, T]
        concat_q_t = torch.stack(concat_q, dim=1)  # [B, T, 4d]

        if train:
            return res_p_t
        else:
            if qtest:
                return res_p_t, concat_q_t
            return res_p_t
