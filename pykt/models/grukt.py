import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        score_path="",
    ):
        super(GRUKT, self).__init__()

        self.pro_max = pro_max
        self.skill_max = skill_max
        self.model_name = "grukt"
        self.emb_type = emb_type
        self.d = d
        self.max_seq = max_seq
        self.use_score_gate = use_score_gate

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
            new_skill = self.skill_gru(x_t, last_skill_state)

            if self.use_score_gate:
                fallback_score = next_ans[:, t].float() * 4.0 + 1.0  # AC->5, non-AC->1
                if next_submission is not None and self.score_table.numel() > 1:
                    sid_t = next_submission[:, t].clamp(min=0)
                    sid_t = sid_t.clamp(max=self.score_table.shape[0] - 1)
                    llm_score = self.score_table[sid_t]
                    score = torch.where(llm_score > 0, llm_score, fallback_score)
                else:
                    score = fallback_score
                alpha = (score / 5.0).unsqueeze(-1).clamp(0.0, 1.0)
                new_skill = alpha * new_skill + (1.0 - alpha) * last_skill_state

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
