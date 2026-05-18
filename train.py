"""
TFT-MPIR 需求预测 — 训练与推理主程序

使用方法
--------
# 1. 用真实数据训练（CSV 需含 dc_id, sku_id, date, sales_qty）：
    python train.py --data path/to/sales_data.csv

# 2. 用模拟数据快速验证：
    python train.py --synthetic

命令行参数见 parse_args()。
"""

import argparse
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

from data_preprocessing import TFTMPIRPreprocessor, make_synthetic_sales
from dataset import build_dataloaders
from tft_model import build_tft_demand_model
from visualize import TrainingVisualizer


# ══════════════════════════════════════════════════════════════════
#  分位数损失
# ══════════════════════════════════════════════════════════════════

def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """
    Pinball / quantile loss。
    pred  : (B, T, Q)
    target: (B, T)
    """
    target = target.unsqueeze(-1).expand_as(pred)   # (B, T, Q)
    q      = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    err    = target - pred
    loss   = torch.max(q * err, (q - 1) * err)
    return loss.mean()


# ══════════════════════════════════════════════════════════════════
#  指标计算
# ══════════════════════════════════════════════════════════════════

def compute_metrics(pred_median: np.ndarray, target: np.ndarray) -> dict:
    """
    使用中位数预测（0.5 分位）计算常用指标。
    pred_median, target: (N,) 反归一化后的真实值域
    """
    mask   = target > 0
    mae    = np.abs(pred_median - target).mean()
    rmse   = np.sqrt(((pred_median - target) ** 2).mean())
    mape   = (np.abs((pred_median[mask] - target[mask]) / target[mask])).mean() * 100
    smape  = (np.abs(pred_median - target) / ((np.abs(pred_median) + np.abs(target)) / 2 + 1e-8)).mean() * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape}


# ══════════════════════════════════════════════════════════════════
#  训练 Epoch
# ══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, quantiles, device) -> float:
    model.train()
    total_loss = 0.0
    for x_enc, x_mark_enc, x_dec, x_mark_dec, y in loader:
        x_enc      = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_dec      = x_dec.to(device)
        x_mark_dec = x_mark_dec.to(device)
        y          = y.to(device)

        optimizer.zero_grad()
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)   # (B, T, Q)
        loss = quantile_loss(pred, y, quantiles)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════
#  评估
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, quantiles, device) -> float:
    model.eval()
    total_loss = 0.0
    for x_enc, x_mark_enc, x_dec, x_mark_dec, y in loader:
        x_enc      = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_dec      = x_dec.to(device)
        x_mark_dec = x_mark_dec.to(device)
        y          = y.to(device)
        pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        total_loss += quantile_loss(pred, y, quantiles).item()
    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════
#  测试推理 + 指标
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_inference(
    model, loader, quantiles, prep: TFTMPIRPreprocessor, device
) -> tuple[dict, dict]:
    """
    返回 (metrics_dict, results_dict)
    results_dict 中包含反归一化后的预测与真值，以及各分位数结果。
    """
    model.eval()
    all_preds  = []   # (N, T, Q)  归一化空间
    all_targets = []  # (N, T)

    for x_enc, x_mark_enc, x_dec, x_mark_dec, y in loader:
        pred = model(
            x_enc.to(device), x_mark_enc.to(device),
            x_dec.to(device), x_mark_dec.to(device)
        ).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y.numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (N, T, Q)
    targets = np.concatenate(all_targets, axis=0)   # (N, T)

    # 取中位数（0.5分位）作为点估计
    mid_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    pred_median = preds[:, :, mid_idx].flatten()
    target_flat = targets.flatten()

    metrics = compute_metrics(pred_median, target_flat)

    results = {
        'pred_quantiles': preds,      # (N, T, Q) 归一化空间
        'target':         targets,    # (N, T)
        'quantiles':      quantiles,
    }
    return metrics, results


# ══════════════════════════════════════════════════════════════════
#  主训练流程
# ══════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"\n{'='*60}")
    print(f"  TFT-MPIR 需求预测训练")
    print(f"  设备: {device}")
    print(f"{'='*60}\n")

    # ── 数据 ─────────────────────────────────────
    if args.synthetic:
        print("使用模拟数据...")
        df = make_synthetic_sales(n_dc=args.n_dc, n_sku=args.n_sku, n_days=args.n_days)
    else:
        print(f"加载数据: {args.data}")
        df = pd.read_csv(args.data, parse_dates=['date'])
        # 字段映射（如有必要）
        col_map = {}
        if args.col_dc:    col_map[args.col_dc]    = 'dc_id'
        if args.col_sku:   col_map[args.col_sku]   = 'sku_id'
        if args.col_date:  col_map[args.col_date]  = 'date'
        if args.col_sales: col_map[args.col_sales] = 'sales_qty'
        if col_map:
            df = df.rename(columns=col_map)

    train_loader, val_loader, test_loader, prep = build_dataloaders(
        df,
        seq_len    = args.seq_len,
        pred_len   = args.pred_len,
        label_len  = 0,
        test_days  = args.test_days,
        val_ratio  = args.val_ratio,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
    )

    # ── 模型 ─────────────────────────────────────
    quantiles = [float(q) for q in args.quantiles.split(',')]
    model = build_tft_demand_model(
        seq_len         = args.seq_len,
        pred_len        = args.pred_len,
        dc_cardinality  = prep.n_dc,
        sku_cardinality = prep.n_sku,
        d_model         = args.d_model,
        n_heads         = args.n_heads,
        quantiles       = quantiles,
        dropout         = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    print(f"预测分位数: {quantiles}\n")

    # ── 优化器 ────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ── 断点续训 ──────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_cnt  = 0
    history       = {'train_loss': [], 'val_loss': [], 'lr': []}
    start_epoch   = 1

    resume_path = os.path.join(args.save_dir, 'last_checkpoint.pt')
    if args.resume and os.path.exists(resume_path):
        print(f"从断点恢复: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        patience_cnt  = ckpt['patience_cnt']
        history       = ckpt['history']
        print(f"  已完成 epoch: {ckpt['epoch']}  最优 val_loss: {best_val_loss:.4f}  "
              f"patience: {patience_cnt}/{args.patience}\n")
    elif args.resume:
        print(f"未找到断点文件 {resume_path}，从头开始训练\n")

    # ── 可视化 ───────────────────────────────────
    viz = TrainingVisualizer(save_dir=args.save_dir)

    # ── 训练循环 ──────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, quantiles, device)
        val_loss   = evaluate(model, val_loader, quantiles, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        viz.update(epoch=epoch, train_loss=train_loss,
                   val_loss=val_loss, lr=current_lr)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={current_lr:.2e}  {elapsed:.1f}s")

        # 每个 epoch 保存 last_checkpoint（覆盖），用于断点续训
        torch.save({
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_cnt':  patience_cnt,
            'history':       history,
            'model_cfg': {
                'seq_len':         args.seq_len,
                'pred_len':        args.pred_len,
                'dc_cardinality':  prep.n_dc,
                'sku_cardinality': prep.n_sku,
                'd_model':         args.d_model,
                'n_heads':         args.n_heads,
                'quantiles':       quantiles,
                'dropout':         args.dropout,
            },
        }, resume_path)

        # 保存最优权重（单独文件，不含优化器状态）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ✓ 最优模型已更新 (val_loss={best_val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n早停触发（patience={args.patience}）")
                break

    viz.save_final()

    # ── 测试评估 ──────────────────────────────────
    print("\n加载最优模型进行测试...")
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pt", map_location=device))
    metrics, results = test_inference(model, test_loader, quantiles, prep, device)

    print("\n" + "="*60)
    print("  测试集评估结果（中位数预测，归一化空间）")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存训练曲线和指标
    def to_py(v):
        if isinstance(v, (np.floating, np.integer)): return float(v)
        if isinstance(v, list): return [to_py(x) for x in v]
        return v
    save_m = {k: to_py(v) for k, v in {**metrics, "best_val_loss": float(best_val_loss), "history": history}.items()}
    with open(f"{args.save_dir}/metrics.json", "w") as f:
        json.dump(save_m, f, indent=2)

    np.save(f"{args.save_dir}/test_predictions.npy", results['pred_quantiles'])
    np.save(f"{args.save_dir}/test_targets.npy",     results['target'])

    print(f"\n模型和结果已保存至: {args.save_dir}/")
    return model, prep, metrics, results


# ══════════════════════════════════════════════════════════════════
#  推理接口（训练完成后使用）
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single_dcsku(
    model:      'TFTDemandForecaster',
    prep:       TFTMPIRPreprocessor,
    history_df: pd.DataFrame,     # 该 DC-SKU 过去 seq_len 天的数据
    dc_id:      str,
    sku_id:     str,
    pred_dates: pd.DatetimeIndex, # 预测的未来日期（pred_len 天）
    device:     torch.device = torch.device('cpu'),
) -> dict:
    """
    对单个 DC-SKU 进行未来 pred_len 天的多分位数预测。

    返回
    ----
    dict: {
        'dates':      DatetimeIndex  预测日期,
        'quantiles':  list[float],
        'predictions': np.ndarray  shape (pred_len, n_quantiles)
    }
    """
    from data_preprocessing import build_time_features

    seq_len  = model.seq_len
    pred_len = model.pred_len

    history_df = history_df.sort_values('date').tail(seq_len)

    key = f'{dc_id}_{sku_id}'
    stat = prep.sales_stats.get(key, {'mean': history_df['sales_qty'].mean(),
                                       'std':  history_df['sales_qty'].std() + 1e-8})
    mu, std = stat['mean'], stat['std']
    sales_norm = (history_df['sales_qty'].values - mu) / std

    dc_code  = prep.dc_vocab.get(dc_id, 0)
    sku_code = prep.sku_vocab.get(sku_id, 0)

    # x_enc
    x_enc = np.stack([
        np.full(seq_len, dc_code,  dtype=np.float32),
        np.full(seq_len, sku_code, dtype=np.float32),
        sales_norm[-seq_len:].astype(np.float32),
    ], axis=-1)[None]   # (1, seq_len, 3)

    enc_dates  = pd.DatetimeIndex(history_df['date'].values[-seq_len:])
    x_mark_enc = build_time_features(enc_dates)[['month','day','weekday','hour']].values[None].astype(np.float32)

    # x_dec（未来已知）
    x_dec      = np.zeros((1, pred_len, 3), dtype=np.float32)
    x_mark_dec = build_time_features(pred_dates)[['month','day','weekday','hour']].values[None].astype(np.float32)

    model.eval()
    out = model(
        torch.from_numpy(x_enc).to(device),
        torch.from_numpy(x_mark_enc).to(device),
        torch.from_numpy(x_dec).to(device),
        torch.from_numpy(x_mark_dec).to(device),
    ).cpu().numpy()[0]   # (pred_len, Q)

    # 反归一化
    pred_denorm = out * std + mu

    return {
        'dates':       pred_dates,
        'quantiles':   model.quantiles,
        'predictions': pred_denorm,   # (pred_len, Q)
    }


# ══════════════════════════════════════════════════════════════════
#  命令行参数
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='TFT-MPIR 需求预测训练')

    # 数据
    p.add_argument('--data',      type=str,  default='',            help='CSV 文件路径')
    p.add_argument('--synthetic', action='store_true',               help='使用模拟数据')
    p.add_argument('--n_dc',      type=int,  default=3)
    p.add_argument('--n_sku',     type=int,  default=5)
    p.add_argument('--n_days',    type=int,  default=300)
    p.add_argument('--col_dc',    type=str,  default='')
    p.add_argument('--col_sku',   type=str,  default='')
    p.add_argument('--col_date',  type=str,  default='')
    p.add_argument('--col_sales', type=str,  default='')

    # 序列
    p.add_argument('--seq_len',   type=int,  default=30)
    p.add_argument('--pred_len',  type=int,  default=7)
    p.add_argument('--test_days', type=int,  default=30)
    p.add_argument('--val_ratio', type=float,default=0.2)

    # 模型
    p.add_argument('--d_model',   type=int,  default=64)
    p.add_argument('--n_heads',   type=int,  default=4)
    p.add_argument('--dropout',   type=float,default=0.1)
    p.add_argument('--quantiles', type=str,  default='0.1,0.5,0.9')

    # 训练
    p.add_argument('--epochs',      type=int,  default=50)
    p.add_argument('--batch_size',  type=int,  default=64)
    p.add_argument('--lr',          type=float,default=0.01)
    p.add_argument('--patience',    type=int,  default=20)
    p.add_argument('--num_workers', type=int,  default=0)
    p.add_argument('--no_cuda',     action='store_true')
    p.add_argument('--resume',      action='store_true',  help='从 last_checkpoint.pt 断点续训')
    p.add_argument('--save_dir',    type=str,  default='./checkpoints')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    args = parse_args()
    if not args.data and not args.synthetic:
        print("未指定数据，使用模拟数据进行演示...")
        args.synthetic = True
    train(args)