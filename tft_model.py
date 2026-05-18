"""
TFT 需求预测模块（对应论文 Fig. 5 需求子模块）

特征布局（与 data_preprocessing.py 对应）：
  x_enc  列：[dc_id(0), sku_id(1), sales(2)]
  static 位置：[0, 1]  → dc_id, sku_id（类别型）
  observed 位置：[2]   → 归一化后的历史销量

时间标记 x_mark：[month, day, weekday, hour]  (4维, embed='fixed')

输出：
  多分位数预测  [B, pred_len, n_quantiles]
  或点预测      [B, pred_len, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════
#  基础模块
# ══════════════════════════════════════════════════════════════════

class GLU(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc1(x) * torch.sigmoid(self.fc2(x))


class GateAddNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.glu        = GLU(input_size, input_size)
        self.projection = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.norm       = nn.LayerNorm(output_size)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        return self.norm(self.projection(self.glu(x) + residual))


class GRN(nn.Module):
    """Gated Residual Network（论文公式核心组件）"""

    def __init__(
        self,
        input_size:   int,
        output_size:  int,
        hidden_size:  Optional[int] = None,
        context_size: Optional[int] = None,
        dropout:      float = 0.0,
    ):
        super().__init__()
        hidden_size = hidden_size or input_size
        self.fc_a    = nn.Linear(input_size, hidden_size)
        self.fc_c    = nn.Linear(context_size, hidden_size) if context_size else None
        self.fc_i    = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.skip    = nn.Linear(input_size, hidden_size) if hidden_size != input_size else nn.Identity()
        self.gate    = GateAddNorm(hidden_size, output_size)

    def forward(self, a: Tensor, c: Optional[Tensor] = None) -> Tensor:
        x = self.fc_a(a)
        if c is not None and self.fc_c is not None:
            # c: (B, d) → 广播到序列维度
            x = x + self.fc_c(c).unsqueeze(1) if a.dim() == 3 else x + self.fc_c(c)
        x = F.elu(x)
        x = self.dropout(self.fc_i(x))
        return self.gate(x, self.skip(a))


# ══════════════════════════════════════════════════════════════════
#  嵌入层
# ══════════════════════════════════════════════════════════════════

class CategoricalEmbedding(nn.Module):
    def __init__(self, cardinality: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.emb     = nn.Embedding(cardinality, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.emb(x.long().clamp(0, self.emb.num_embeddings - 1)))


class NumericEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.proj    = nn.Linear(1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(x.unsqueeze(-1)))


class TemporalEmbedding(nn.Module):
    """
    固定时间嵌入（embed='fixed'）
    输入 x_mark: (B, T, 4)  → [month, day, weekday, hour]
    输出: (B, T, n_time_feat, d_model)  逐特征独立嵌入后 stack
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.month_emb   = nn.Embedding(12, d_model)
        self.day_emb     = nn.Embedding(31, d_model)
        self.weekday_emb = nn.Embedding(7,  d_model)
        self.hour_emb    = nn.Embedding(24, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        month   = self.month_emb(x[..., 0])    # (B,T,d)
        day     = self.day_emb(x[..., 1])
        weekday = self.weekday_emb(x[..., 2])
        hour    = self.hour_emb(x[..., 3])
        return torch.stack([month, day, weekday, hour], dim=-2)   # (B,T,4,d)


# ══════════════════════════════════════════════════════════════════
#  变量选择网络 VSN
# ══════════════════════════════════════════════════════════════════

class VariableSelectionNetwork(nn.Module):
    """
    输入 x: (B, T, n_vars, d_model) 或 (B, n_vars, d_model)
    输出:  (B, T, d_model)          或 (B, d_model)
    """

    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.0):
        super().__init__()
        self.joint_grn = GRN(
            d_model * n_vars, n_vars,
            hidden_size=d_model, context_size=d_model, dropout=dropout
        )
        self.var_grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(n_vars)])

    def forward(self, x: Tensor, ctx: Optional[Tensor] = None) -> Tensor:
        flat    = x.flatten(start_dim=-2)                              # (..., n_vars*d)
        weights = F.softmax(self.joint_grn(flat, ctx), dim=-1)        # (..., n_vars)
        processed = torch.stack([grn(x[..., i, :]) for i, grn in enumerate(self.var_grns)], dim=-1)
        return torch.matmul(processed, weights.unsqueeze(-1)).squeeze(-1)  # (..., d)


# ══════════════════════════════════════════════════════════════════
#  静态编码器
# ══════════════════════════════════════════════════════════════════

class StaticCovariateEncoder(nn.Module):
    """输出 4 个静态 context 向量：c_s, c_c, c_h, c_e"""

    def __init__(self, d_model: int, n_static: int, dropout: float = 0.0):
        super().__init__()
        self.vsn  = VariableSelectionNetwork(d_model, n_static, dropout=dropout)
        self.grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(4)])

    def forward(self, static_emb: Tensor) -> List[Tensor]:
        # static_emb: (B, n_static, d_model)
        feat = self.vsn(static_emb)          # (B, d_model)
        return [grn(feat) for grn in self.grns]   # 4 × (B, d_model)


# ══════════════════════════════════════════════════════════════════
#  可解释多头自注意力
# ══════════════════════════════════════════════════════════════════

class InterpretableMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        # Q,K 各一套头；V 共享（可解释性设计）
        self.qk_proj = nn.Linear(d_model, 2 * n_heads * self.d_head, bias=False)
        self.v_proj  = nn.Linear(d_model, self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)
        self.scale    = self.d_head ** -0.5
        # Causal mask（动态生成，支持任意长度）
        self._max_len = max_len

    def _get_mask(self, T: int, device) -> Tensor:
        return torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        qk = self.qk_proj(x)                                      # (B,T, 2*n*d_h)
        q, k = qk.chunk(2, dim=-1)                                 # each (B,T, n*d_h)
        v = self.v_proj(x)                                          # (B,T, d_h)

        q = q.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)   # (B,n,T,d_h)
        k = k.view(B, T, self.n_heads, self.d_head).permute(0, 2, 3, 1)   # (B,n,d_h,T)

        scores = torch.matmul(q, k) * self.scale                   # (B,n,T,T)
        scores = scores + self._get_mask(T, x.device)
        attn   = F.softmax(scores, dim=-1)                          # (B,n,T,T)

        v_exp  = v.unsqueeze(1)                                     # (B,1,T,d_h)
        out    = torch.matmul(attn, v_exp).mean(dim=1)             # (B,T,d_h)
        return self.dropout(self.out_proj(out))                     # (B,T,d_model)


# ══════════════════════════════════════════════════════════════════
#  TFT 需求预测模型
# ══════════════════════════════════════════════════════════════════

class TFTDemandForecaster(nn.Module):
    """
    论文 Fig. 5 需求子模块的完整实现。

    参数
    ----
    seq_len        : 历史回看窗口（T̃）
    pred_len       : 预测步数（τ_max）
    d_model        : 隐层维度
    n_heads        : 注意力头数
    n_static       : 静态变量数（dc + sku = 2）
    n_observed     : 历史观测变量数（sales = 1）
    n_known        : 已知未来时间特征数（month/day/weekday/hour = 4）
    dc_cardinality : DC 类别数
    sku_cardinality: SKU 类别数
    quantiles      : 预测分位数列表，为空则做点预测
    dropout        : Dropout 率
    """

    def __init__(
        self,
        seq_len:         int,
        pred_len:        int,
        d_model:         int = 64,
        n_heads:         int = 4,
        n_static:        int = 2,
        n_observed:      int = 1,
        n_known:         int = 4,
        dc_cardinality:  int = 18,
        sku_cardinality: int = 72,
        quantiles:       List[float] = (0.1, 0.5, 0.9),
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.d_model   = d_model
        self.quantiles = list(quantiles)
        self.n_known   = n_known
        self.n_observed = n_observed
        self.n_static  = n_static

        output_size = len(quantiles) if quantiles else 1

        # ── 嵌入层 ──────────────────────────────
        # 静态：dc(类别) + sku(类别)
        self.dc_emb  = CategoricalEmbedding(dc_cardinality,  d_model, dropout)
        self.sku_emb = CategoricalEmbedding(sku_cardinality, d_model, dropout)

        # 观测：历史销量（数值）
        self.sales_emb = NumericEmbedding(d_model, dropout)

        # 已知未来：时间特征（固定嵌入）
        self.time_emb = TemporalEmbedding(d_model)

        # ── 编码器 ───────────────────────────────
        self.static_encoder = StaticCovariateEncoder(d_model, n_static, dropout)

        # history VSN: observed(1) + known_hist(4)
        self.history_vsn = VariableSelectionNetwork(d_model, n_observed + n_known, dropout)
        # future VSN: known_future(4)
        self.future_vsn  = VariableSelectionNetwork(d_model, n_known, dropout)

        # ── 时序解码器 ───────────────────────────
        self.hist_lstm  = nn.LSTM(d_model, d_model, batch_first=True)
        self.fut_lstm   = nn.LSTM(d_model, d_model, batch_first=True)
        self.gate_lstm  = GateAddNorm(d_model, d_model)

        self.enrich_grn = GRN(d_model, d_model, context_size=d_model, dropout=dropout)
        self.attention  = InterpretableMultiHeadAttention(d_model, n_heads, seq_len + pred_len, dropout)
        self.gate_attn  = GateAddNorm(d_model, d_model)

        self.pos_grn    = GRN(d_model, d_model, dropout=dropout)
        self.gate_final = GateAddNorm(d_model, d_model)

        self.out_proj   = nn.Linear(d_model, output_size)

    # ── 前向传播 ─────────────────────────────────
    def forward(
        self,
        x_enc:      Tensor,   # (B, seq_len, 3)  [dc, sku, sales]
        x_mark_enc: Tensor,   # (B, seq_len, 4)  [month, day, weekday, hour]
        x_dec:      Tensor,   # (B, pred_len, 3) （解码器，sales 部分为 0）
        x_mark_dec: Tensor,   # (B, pred_len, 4)
    ) -> Tensor:
        B = x_enc.size(0)

        # ── 静态嵌入 ──────────────────────────────
        dc_e  = self.dc_emb(x_enc[:, 0, 0])    # (B, d)  取第一个时间步的静态值
        sku_e = self.sku_emb(x_enc[:, 0, 1])
        static_emb = torch.stack([dc_e, sku_e], dim=1)  # (B, 2, d)

        c_s, c_c, c_h, c_e = self.static_encoder(static_emb)

        # ── 时间嵌入（encoder + decoder 拼接）───────
        x_mark_full = torch.cat([x_mark_enc, x_mark_dec], dim=1)   # (B, T+τ, 4)
        time_emb    = self.time_emb(x_mark_full)                    # (B, T+τ, 4, d)
        time_enc    = time_emb[:, :self.seq_len]                    # (B, T,  4, d)
        time_dec    = time_emb[:, self.seq_len:]                    # (B, τ,  4, d)

        # ── 历史观测嵌入 ─────────────────────────────
        sales_emb = self.sales_emb(x_enc[:, :, 2])                  # (B, T, d)

        # history VSN 输入: [sales] + [time×4] → (B, T, 5, d)
        hist_vars = torch.cat([
            sales_emb.unsqueeze(-2),   # (B, T, 1, d)
            time_enc,                  # (B, T, 4, d)
        ], dim=-2)                     # (B, T, 5, d)

        # future VSN 输入: [time×4] → (B, τ, 4, d)
        fut_vars = time_dec            # (B, τ, 4, d)

        hist_selected = self.history_vsn(hist_vars, c_s)  # (B, T, d)
        fut_selected  = self.future_vsn(fut_vars,  c_s)   # (B, τ, d)

        # ── LSTM ─────────────────────────────────────
        h0 = c_h.unsqueeze(0)  # (1, B, d)
        c0 = c_c.unsqueeze(0)
        hist_out, (hn, cn) = self.hist_lstm(hist_selected, (h0, c0))
        fut_out,  _        = self.fut_lstm(fut_selected,  (hn, cn))

        # skip connection
        temporal_input = torch.cat([hist_selected, fut_selected], dim=1)  # (B, T+τ, d)
        temporal_feat  = torch.cat([hist_out, fut_out], dim=1)            # (B, T+τ, d)
        temporal_feat  = self.gate_lstm(temporal_feat, temporal_input)

        # ── 静态增强 ──────────────────────────────────
        enriched = self.enrich_grn(temporal_feat, c_e)   # (B, T+τ, d)

        # ── 自注意力（因果）──────────────────────────
        attn_out = self.attention(enriched)               # (B, T+τ, d)
        # 仅对预测步做输出，跳过历史部分
        attn_out = self.gate_attn(
            attn_out[:, -self.pred_len:],
            enriched[:, -self.pred_len:]
        )                                                  # (B, τ, d)

        # ── 位置级 FFN + 最终门控 ────────────────────
        out = self.pos_grn(attn_out)
        out = self.gate_final(out, temporal_feat[:, -self.pred_len:])

        return self.out_proj(out)   # (B, τ, Q)  Q=len(quantiles) or 1


# ══════════════════════════════════════════════════════════════════
#  工厂函数
# ══════════════════════════════════════════════════════════════════

def build_tft_demand_model(
    seq_len:         int,
    pred_len:        int,
    dc_cardinality:  int,
    sku_cardinality: int,
    d_model:         int = 64,
    n_heads:         int = 4,
    quantiles:       List[float] = (0.1, 0.5, 0.9),
    dropout:         float = 0.1,
) -> TFTDemandForecaster:
    return TFTDemandForecaster(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=d_model,
        n_heads=n_heads,
        n_static=2,       # dc + sku
        n_observed=1,     # sales
        n_known=4,        # month/day/weekday/hour
        dc_cardinality=dc_cardinality,
        sku_cardinality=sku_cardinality,
        quantiles=list(quantiles),
        dropout=dropout,
    )


# ══════════════════════════════════════════════════════════════════
#  快速测试
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    model = build_tft_demand_model(
        seq_len=30, pred_len=7,
        dc_cardinality=18, sku_cardinality=72,
    )
    B = 4
    x_enc      = torch.zeros(B, 30, 3)
    x_mark_enc = torch.zeros(B, 30, 4).long()
    x_dec      = torch.zeros(B, 7,  3)
    x_mark_dec = torch.zeros(B, 7,  4).long()

    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print(f"模型输出形状: {out.shape}")   # 期望 (4, 7, 3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数总量: {n_params:,}")
