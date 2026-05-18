"""
TFT-MPIR 数据预处理模块
对应论文 Section 3.2 的输入特征构造：
  - static:   DC编号(类别), SKU编号(类别)
  - observed: 历史销量 (日销量 target)
  - known:    月/日/星期/是否周末/是否周一或周五(补货日)

输入 CSV 格式（参考 data_exp.md）:
  dc_id, sku_id, date, sales_qty
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  日历特征
# ─────────────────────────────────────────────

def build_time_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    生成论文中 z_t (known future inputs) 对应的时间特征。
    返回 DataFrame，列顺序与 TFTTimeFeatureEmbedding / TFTTemporalEmbedding 一致：
      [month, day, weekday, (hour=0)]
    额外附加：
      is_weekend, is_replenishment_day (周一/周五)
    """
    df = pd.DataFrame(index=dates)
    df['month']    = dates.month - 1          # 0-11
    df['day']      = dates.day - 1            # 0-30
    df['weekday']  = dates.weekday            # 0=Mon … 6=Sun
    df['hour']     = 0                        # 日粒度，hour 恒为 0
    # 语义特征（供后续分析用，不直接进 TFT embed 的时间编码）
    df['is_weekend']          = (dates.weekday >= 5).astype(int)
    df['is_replenishment_day'] = dates.weekday.isin([0, 4]).astype(int)  # Mon/Fri
    return df


# ─────────────────────────────────────────────
#  主预处理类
# ─────────────────────────────────────────────

class TFTMPIRPreprocessor:
    """
    将原始销售 CSV 转换为 TFT 训练所需的序列样本。

    参数
    ----
    seq_len   : 历史回看窗口 T̃（论文取 30）
    pred_len  : 预测步数 τ_max（默认 30，与测试集对齐）
    label_len : decoder 中历史片段长度（TFT 默认 = seq_len）
    """

    def __init__(self, seq_len: int = 30, pred_len: int = 30, label_len: int = 0):
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.label_len = label_len

        # 将在 fit() 中确定
        self.dc_vocab:  dict[str, int] = {}
        self.sku_vocab: dict[str, int] = {}
        self.sales_stats: dict[str, dict] = {}  # 每个 DC-SKU 的均值/标准差

    # ── fit ────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        """
        df 必须含列: dc_id, sku_id, date, sales_qty
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 编码 DC / SKU
        dc_ids  = sorted(df['dc_id'].unique())
        sku_ids = sorted(df['sku_id'].unique())
        self.dc_vocab  = {v: i for i, v in enumerate(dc_ids)}
        self.sku_vocab = {v: i for i, v in enumerate(sku_ids)}

        # 仅在训练集上计算归一化统计量
        for (dc, sku), grp in df.groupby(['dc_id', 'sku_id']):
            key = f'{dc}_{sku}'
            mu  = grp['sales_qty'].mean()
            std = grp['sales_qty'].std() + 1e-8
            self.sales_stats[key] = {'mean': mu, 'std': std}

        return self

    # ── transform ─────────────────────────────
    def transform(self, df: pd.DataFrame) -> list[dict]:
        """
        返回样本列表，每个样本是一个 dict：
          x_enc      : [seq_len, n_features]  encoder 输入（observed + static 拼合）
          x_mark_enc : [seq_len, 4]           encoder 时间标记 [month,day,weekday,hour]
          x_dec      : [label_len+pred_len, n_features]
          x_mark_dec : [label_len+pred_len, 4]
          y          : [pred_len]             目标销量（归一化后）
          meta       : dict  {dc_id, sku_id, date_start, date_end}
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['dc_id', 'sku_id', 'date'])

        samples = []
        total   = self.seq_len + self.pred_len

        for (dc, sku), grp in df.groupby(['dc_id', 'sku_id']):
            grp = grp.sort_values('date').reset_index(drop=True)
            if len(grp) < total:
                continue  # 数据不足，跳过

            key     = f'{dc}_{sku}'
            mu      = self.sales_stats.get(key, {}).get('mean', grp['sales_qty'].mean())
            std     = self.sales_stats.get(key, {}).get('std', grp['sales_qty'].std() + 1e-8)

            sales_norm = (grp['sales_qty'].values - mu) / std

            dc_code  = self.dc_vocab.get(dc, 0)
            sku_code = self.sku_vocab.get(sku, 0)

            dates     = pd.DatetimeIndex(grp['date'])
            time_feat = build_time_features(dates)
            time_arr  = time_feat[['month', 'day', 'weekday', 'hour']].values  # (T, 4)

            # 滑窗切片
            for i in range(total, len(grp) + 1):
                enc_slice = slice(i - total, i - self.pred_len)
                dec_slice = slice(i - self.pred_len - self.label_len, i)

                # encoder 输入: [seq_len, 1(sales) + 2(static)]
                # static 特征在每个时间步重复（TFTEmbedding 内部会取 x_enc[:,0,pos] 作为静态）
                enc_sales  = sales_norm[enc_slice]                           # (seq_len,)
                enc_static = np.array([dc_code, sku_code], dtype=np.float32)  # (2,)

                # 拼为 x_enc: shape (seq_len, 3)  列: [dc_code, sku_code, sales]
                x_enc = np.stack([
                    np.full(self.seq_len, dc_code,  dtype=np.float32),
                    np.full(self.seq_len, sku_code, dtype=np.float32),
                    enc_sales.astype(np.float32),
                ], axis=-1)

                x_mark_enc = time_arr[enc_slice].astype(np.float32)

                # decoder 输入: label_len 历史 + pred_len 零（TFT 仅用 known future）
                dec_len = self.label_len + self.pred_len
                x_dec = np.zeros((dec_len, 3), dtype=np.float32)
                if self.label_len > 0:
                    x_dec[:self.label_len] = x_enc[-self.label_len:]

                x_mark_dec = time_arr[dec_slice].astype(np.float32)

                y = sales_norm[i - self.pred_len: i].astype(np.float32)

                samples.append({
                    'x_enc':      x_enc,
                    'x_mark_enc': x_mark_enc,
                    'x_dec':      x_dec,
                    'x_mark_dec': x_mark_dec,
                    'y':          y,
                    'meta': {
                        'dc_id':      dc,
                        'sku_id':     sku,
                        'mean':       mu,
                        'std':        std,
                        'date_start': grp['date'].iloc[i - total],
                        'date_end':   grp['date'].iloc[i - 1],
                    }
                })

        return samples

    def fit_transform(self, df: pd.DataFrame) -> list[dict]:
        return self.fit(df).transform(df)

    # ── 逆归一化 ──────────────────────────────
    def inverse_transform_sales(self, y_norm: np.ndarray, dc_id, sku_id) -> np.ndarray:
        key  = f'{dc_id}_{sku_id}'
        stat = self.sales_stats.get(key, {'mean': 0, 'std': 1})
        return y_norm * stat['std'] + stat['mean']

    @property
    def n_dc(self):
        return len(self.dc_vocab)

    @property
    def n_sku(self):
        return len(self.sku_vocab)


# ─────────────────────────────────────────────
#  构造模拟数据（用于单元测试）
# ─────────────────────────────────────────────

def make_synthetic_sales(
    n_dc: int = 3,
    n_sku: int = 4,
    n_days: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """生成用于测试的模拟销售数据。"""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range('2018-01-01', periods=n_days, freq='D')
    rows  = []
    for dc in range(n_dc):
        for sku in range(n_sku):
            base  = rng.uniform(10, 200)
            trend = rng.uniform(-0.01, 0.05)
            seas  = rng.uniform(0, 0.3)
            noise = rng.uniform(0.05, 0.2)
            t     = np.arange(n_days)
            sales = (
                base
                + base * trend * t
                + base * seas * np.sin(2 * np.pi * t / 7)    # 周季节性
                + base * seas * np.sin(2 * np.pi * t / 365)  # 年季节性
                + rng.normal(0, base * noise, n_days)
            ).clip(0).round()
            for i, d in enumerate(dates):
                rows.append({'dc_id': f'DC{dc:02d}', 'sku_id': f'SKU{sku:03d}',
                             'date': d, 'sales_qty': sales[i]})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  快速测试
# ─────────────────────────────────────────────

if __name__ == '__main__':
    df = make_synthetic_sales(n_dc=2, n_sku=3, n_days=120)
    prep = TFTMPIRPreprocessor(seq_len=30, pred_len=7)
    samples = prep.fit_transform(df)
    print(f'生成样本数: {len(samples)}')
    s = samples[0]
    print(f"x_enc shape:      {s['x_enc'].shape}")
    print(f"x_mark_enc shape: {s['x_mark_enc'].shape}")
    print(f"x_dec shape:      {s['x_dec'].shape}")
    print(f"x_mark_dec shape: {s['x_mark_dec'].shape}")
    print(f"y shape:          {s['y'].shape}")
    print(f"DC词表大小: {prep.n_dc}, SKU词表大小: {prep.n_sku}")
