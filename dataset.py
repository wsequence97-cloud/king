"""
TFT-MPIR Dataset / DataLoader 模块
将 TFTMPIRPreprocessor 生成的样本列表封装为 PyTorch Dataset。
支持训练集 / 验证集 / 测试集的标准切分逻辑（论文：最后30天为测试集）。
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional

from data_preprocessing import TFTMPIRPreprocessor, make_synthetic_sales


# ─────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────

class TFTDataset(Dataset):
    """
    封装预处理后的样本列表为 PyTorch Dataset。

    返回的每个样本 (x_enc, x_mark_enc, x_dec, x_mark_dec, y)
    均为 torch.FloatTensor，可直接送入模型。
    """

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s['x_enc']),
            torch.from_numpy(s['x_mark_enc']),
            torch.from_numpy(s['x_dec']),
            torch.from_numpy(s['x_mark_dec']),
            torch.from_numpy(s['y']),
        )


# ─────────────────────────────────────────────
#  数据切分
# ─────────────────────────────────────────────

def split_dataframe_by_time(
    df: pd.DataFrame,
    test_days: int = 30,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按论文 Section 4.2 规则切分数据：
      - 每个 DC-SKU 的最后 test_days 天作为测试集
      - 剩余数据按 val_ratio 切分为训练集和验证集

    返回 (train_df, val_df, test_df)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    train_rows, val_rows, test_rows = [], [], []

    for _, grp in df.groupby(['dc_id', 'sku_id']):
        grp = grp.sort_values('date')
        n   = len(grp)

        if n <= test_days:
            # 数据量不足，整体归入训练集
            train_rows.append(grp)
            continue

        test_part = grp.iloc[-test_days:]
        rest_part = grp.iloc[:-test_days]

        val_size  = max(1, int(len(rest_part) * val_ratio))
        val_part  = rest_part.iloc[-val_size:]
        train_part = rest_part.iloc[:-val_size]

        train_rows.append(train_part)
        val_rows.append(val_part)
        test_rows.append(test_part)

    train_df = pd.concat(train_rows).reset_index(drop=True) if train_rows else pd.DataFrame()
    val_df   = pd.concat(val_rows).reset_index(drop=True)   if val_rows   else pd.DataFrame()
    test_df  = pd.concat(test_rows).reset_index(drop=True)  if test_rows  else pd.DataFrame()

    return train_df, val_df, test_df


# ─────────────────────────────────────────────
#  一站式构建函数
# ─────────────────────────────────────────────

def build_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 30,
    pred_len: int = 7,
    label_len: int = 0,
    test_days: int = 30,
    val_ratio: float = 0.2,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, TFTMPIRPreprocessor]:
    """
    一站式函数：给定原始 DataFrame，返回 (train_loader, val_loader, test_loader, preprocessor)。

    注意
    ----
    preprocessor 仅在训练集上 fit，避免数据泄露。
    验证集 / 测试集同样使用训练集计算的归一化统计量，
    但时间窗口会包含验证/测试期的实际销量（作为 encoder 的历史输入）。
    """
    train_df, val_df, test_df = split_dataframe_by_time(df, test_days, val_ratio)

    prep = TFTMPIRPreprocessor(seq_len=seq_len, pred_len=pred_len, label_len=label_len)
    prep.fit(train_df)

    # 构造样本时需要完整时间序列（encoder 需要历史数据）
    # 因此验证集使用 train+val，测试集使用 train+val+test 的完整序列
    full_train_samples = prep.transform(train_df)
    full_val_samples   = prep.transform(pd.concat([train_df, val_df]).sort_values(['dc_id','sku_id','date']))
    full_test_samples  = prep.transform(pd.concat([train_df, val_df, test_df]).sort_values(['dc_id','sku_id','date']))

    # 验证集：仅取窗口末端落在 val 期间的样本
    val_start_dates = val_df.groupby(['dc_id','sku_id'])['date'].min().to_dict()
    val_samples = [
        s for s in full_val_samples
        if s['meta']['date_end'] >= val_start_dates.get(
            (s['meta']['dc_id'], s['meta']['sku_id']),
            pd.Timestamp('2099-01-01')
        )
    ]

    test_start_dates = test_df.groupby(['dc_id','sku_id'])['date'].min().to_dict()
    test_samples = [
        s for s in full_test_samples
        if s['meta']['date_end'] >= test_start_dates.get(
            (s['meta']['dc_id'], s['meta']['sku_id']),
            pd.Timestamp('2099-01-01')
        )
    ]

    train_loader = DataLoader(TFTDataset(full_train_samples), batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(TFTDataset(val_samples),        batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(TFTDataset(test_samples),       batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"数据集规模 — 训练: {len(full_train_samples)}  验证: {len(val_samples)}  测试: {len(test_samples)}")
    return train_loader, val_loader, test_loader, prep


# ─────────────────────────────────────────────
#  快速测试
# ─────────────────────────────────────────────

if __name__ == '__main__':
    df = make_synthetic_sales(n_dc=3, n_sku=4, n_days=200)
    train_loader, val_loader, test_loader, prep = build_dataloaders(
        df, seq_len=30, pred_len=7, batch_size=32
    )
    for batch in train_loader:
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
        print("训练批次形状:")
        print(f"  x_enc:      {x_enc.shape}")        # (B, seq_len, 3)
        print(f"  x_mark_enc: {x_mark_enc.shape}")   # (B, seq_len, 4)
        print(f"  x_dec:      {x_dec.shape}")         # (B, pred_len, 3)
        print(f"  x_mark_dec: {x_mark_dec.shape}")    # (B, pred_len, 4)
        print(f"  y:          {y.shape}")              # (B, pred_len)
        break
