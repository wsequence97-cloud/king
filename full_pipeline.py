import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from tft_model import build_tft_demand_model


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_date(df: pd.DataFrame, date_col: str, val_ratio: float = 0.15, test_ratio: float = 0.15):
    dates = np.array(sorted(pd.to_datetime(df[date_col]).unique()))
    n_dates = len(dates)
    test_size = max(1, int(n_dates * test_ratio))
    val_size = max(1, int(n_dates * val_ratio))
    test_start = dates[-test_size]
    val_start = dates[-(test_size + val_size)]
    train_df = df[pd.to_datetime(df[date_col]) < val_start].copy()
    val_df = df[(pd.to_datetime(df[date_col]) >= val_start) & (pd.to_datetime(df[date_col]) < test_start)].copy()
    test_df = df[pd.to_datetime(df[date_col]) >= test_start].copy()
    return train_df, val_df, test_df


def split_sales_by_time(
    df: pd.DataFrame,
    test_days: int = 30,
    val_ratio: float = 0.2,
):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train_rows, val_rows, test_rows = [], [], []
    for _, grp in df.groupby(["dc_id", "sku_id"]):
        grp = grp.sort_values("date")
        n = len(grp)
        if n <= test_days:
            train_rows.append(grp)
            continue
        test_part = grp.iloc[-test_days:]
        rest_part = grp.iloc[:-test_days]
        val_size = max(1, int(len(rest_part) * val_ratio))
        val_part = rest_part.iloc[-val_size:]
        train_part = rest_part.iloc[:-val_size]
        train_rows.append(train_part)
        val_rows.append(val_part)
        test_rows.append(test_part)
    train_df = pd.concat(train_rows).reset_index(drop=True) if train_rows else pd.DataFrame()
    val_df = pd.concat(val_rows).reset_index(drop=True) if val_rows else pd.DataFrame()
    test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame()
    return train_df, val_df, test_df


def build_time_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    feat = pd.DataFrame(index=dates)
    feat["month"] = dates.month - 1
    feat["day"] = dates.day - 1
    feat["weekday"] = dates.weekday
    feat["hour"] = 0
    feat["is_weekend"] = (dates.weekday >= 5).astype(int)
    feat["is_replenishment_day"] = dates.weekday.isin([0, 4]).astype(int)
    return feat


class TFTDemandAdapter:
    def __init__(self, seq_len: int, pred_len: int):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dc_vocab = {}
        self.sku_vocab = {}
        self.sales_stats = {}

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        dc_ids = sorted(df["dc_id"].unique())
        sku_ids = sorted(df["sku_id"].unique())
        self.dc_vocab = {v: i for i, v in enumerate(dc_ids)}
        self.sku_vocab = {v: i for i, v in enumerate(sku_ids)}
        for (dc, sku), grp in df.groupby(["dc_id", "sku_id"]):
            values = grp["sales_qty"].astype(float)
            self.sales_stats["{}_{}".format(dc, sku)] = {
                "mean": float(values.mean()),
                "std": float(values.std() + 1e-8),
            }
        return self


def compute_regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mask = np.abs(target) > 1e-6
    mape = float(np.mean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100) if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class LeadTimeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        pred = self.head(out[:, -1, :]).squeeze(-1)
        return pred


class ReplenishmentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).view(-1)
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    model.eval()
    total = 0.0
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).view(-1)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        total += loss.item()
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])
    return total / max(1, len(loader)), preds, targets


def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = evaluate_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def load_reference_tables(data_dir: Path):
    sales = pd.read_csv(data_dir / "sales_data.csv")
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales.rename(columns={"sale": "sales_qty"})

    lead = pd.read_csv(data_dir / "leadtime_data.csv")
    lead["date"] = pd.to_datetime(lead["date"])
    if "idx" in lead.columns:
        split_vals = lead["idx"].str.split("_", expand=True)
        lead["dc_id"] = split_vals[0]
        lead["sku_id"] = split_vals[1]

    dc_inventory = pd.read_csv(data_dir / "dc_inventory.csv")
    dc_inventory["available_inv"] = pd.to_numeric(dc_inventory["available_inv"], errors="coerce").fillna(0.0)

    factory_inventory = pd.read_csv(data_dir / "factory_inventory.csv")
    factory_inventory["available_inv"] = pd.to_numeric(factory_inventory["available_inv"], errors="coerce").fillna(0.0)

    dc_capacity = pd.read_csv(data_dir / "dc_capacity.csv")
    dc_capacity["capacity"] = pd.to_numeric(dc_capacity["capacity"], errors="coerce").fillna(0.0)

    push_limit = pd.read_csv(data_dir / "push_limit.csv")
    push_limit["push_lb"] = pd.to_numeric(push_limit["push_lb"], errors="coerce").fillna(0.0)
    push_limit["push_ub"] = pd.to_numeric(push_limit["push_ub"], errors="coerce").fillna(0.0)

    tariff = pd.read_csv(data_dir / "transport_tariff.csv")
    tariff["unit_cost"] = pd.to_numeric(tariff["unit_cost"], errors="coerce").fillna(0.0)

    unit_rate = pd.read_csv(data_dir / "unit_rate.csv")
    unit_rate["pt_box"] = pd.to_numeric(unit_rate["pt_box"], errors="coerce").fillna(1.0)
    unit_rate["box_volume"] = pd.to_numeric(unit_rate["box_volume"], errors="coerce").fillna(1.0)
    return sales, lead, dc_inventory, factory_inventory, dc_capacity, push_limit, tariff, unit_rate


def build_vlt_samples(lead_df: pd.DataFrame, seq_len: int = 14):
    samples = []
    meta = []
    for (dc_id, sku_id), grp in lead_df.groupby(["dc_id", "sku_id"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["weekday"] = grp["date"].dt.weekday
        grp["month"] = grp["date"].dt.month
        grp["is_weekend"] = (grp["weekday"] >= 5).astype(float)
        values = grp["leadtime"].astype(float).values
        for end in range(seq_len, len(grp)):
            hist = values[end - seq_len:end]
            hist_dates = grp.iloc[end - seq_len:end]
            x = np.stack(
                [
                    hist,
                    hist_dates["weekday"].values.astype(float),
                    hist_dates["month"].values.astype(float),
                    hist_dates["is_weekend"].values.astype(float),
                ],
                axis=-1,
            )
            y = values[end]
            samples.append(x.astype(np.float32))
            meta.append(
                {
                    "dc_id": dc_id,
                    "sku_id": sku_id,
                    "date": grp.loc[end, "date"],
                    "target": float(y),
                }
            )
    x = np.array(samples, dtype=np.float32)
    y = np.array([m["target"] for m in meta], dtype=np.float32)
    meta_df = pd.DataFrame(meta)
    return x, y, meta_df


def rolling_demand_forecast(history_sales: np.ndarray, horizon: int) -> float:
    if len(history_sales) == 0:
        return 0.0
    recent_7 = history_sales[-7:] if len(history_sales) >= 7 else history_sales
    recent_30 = history_sales[-30:] if len(history_sales) >= 30 else history_sales
    daily_base = 0.7 * recent_7.mean() + 0.3 * recent_30.mean()
    return float(max(0.0, daily_base * horizon))


def load_tft_checkpoint(checkpoint_dir: Path, device: torch.device):
    resume_path = checkpoint_dir / "last_checkpoint.pt"
    best_path = checkpoint_dir / "best_model.pt"
    ckpt = torch.load(resume_path, map_location=device)
    cfg = ckpt["model_cfg"]
    model = build_tft_demand_model(
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        dc_cardinality=cfg["dc_cardinality"],
        sku_cardinality=cfg["sku_cardinality"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        quantiles=cfg["quantiles"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    return model, cfg


def sum_forecast_horizon(pred_seq: np.ndarray, horizon: int) -> float:
    if pred_seq.size == 0:
        return 0.0
    horizon = max(1, int(horizon))
    if horizon <= len(pred_seq):
        return float(np.maximum(pred_seq[:horizon], 0.0).sum())
    extension = np.full(horizon - len(pred_seq), max(float(pred_seq[-1]), 0.0), dtype=np.float32)
    full = np.concatenate([np.maximum(pred_seq, 0.0), extension])
    return float(full.sum())


def build_tft_demand_prediction_frame(
    sales: pd.DataFrame,
    lead: pd.DataFrame,
    checkpoint_dir: Path,
    device: torch.device,
    batch_size: int = 512,
    test_days: int = 30,
    val_ratio: float = 0.2,
):
    model, cfg = load_tft_checkpoint(checkpoint_dir, device)
    seq_len = int(cfg["seq_len"])
    pred_len = int(cfg["pred_len"])
    quantiles = list(cfg["quantiles"])
    mid_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    train_sales, _, _ = split_sales_by_time(sales, test_days=test_days, val_ratio=val_ratio)
    prep = TFTDemandAdapter(seq_len=seq_len, pred_len=pred_len)
    prep.fit(train_sales)

    sales = sales.sort_values(["dc_id", "sku_id", "date"]).reset_index(drop=True)
    sales_group = {k: v.sort_values("date").reset_index(drop=True) for k, v in sales.groupby(["dc_id", "sku_id"])}

    demand_requests = []
    x_enc_list, x_mark_enc_list, x_dec_list, x_mark_dec_list = [], [], [], []

    for row in lead.itertuples(index=False):
        key = (row.dc_id, row.sku_id)
        grp = sales_group.get(key)
        if grp is None:
            continue
        current_date = pd.Timestamp(row.date)
        history = grp[grp["date"] <= current_date].tail(seq_len)
        if len(history) < seq_len:
            continue

        stat_key = f"{row.dc_id}_{row.sku_id}"
        stat = prep.sales_stats.get(
            stat_key,
            {"mean": float(history["sales_qty"].mean()), "std": float(history["sales_qty"].std(ddof=0) + 1e-8)},
        )
        mu = stat["mean"]
        std = stat["std"] if stat["std"] > 0 else 1.0
        sales_norm = ((history["sales_qty"].values.astype(np.float32) - mu) / std).astype(np.float32)
        dc_code = prep.dc_vocab.get(row.dc_id, 0)
        sku_code = prep.sku_vocab.get(row.sku_id, 0)

        x_enc = np.stack(
            [
                np.full(seq_len, dc_code, dtype=np.float32),
                np.full(seq_len, sku_code, dtype=np.float32),
                sales_norm,
            ],
            axis=-1,
        )
        x_mark_enc = build_time_features(pd.DatetimeIndex(history["date"]))[["month", "day", "weekday", "hour"]].values.astype(np.float32)
        pred_dates = pd.date_range(current_date + pd.Timedelta(days=1), periods=pred_len, freq="D")
        x_dec = np.zeros((pred_len, 3), dtype=np.float32)
        x_mark_dec = build_time_features(pred_dates)[["month", "day", "weekday", "hour"]].values.astype(np.float32)

        x_enc_list.append(x_enc)
        x_mark_enc_list.append(x_mark_enc)
        x_dec_list.append(x_dec)
        x_mark_dec_list.append(x_mark_dec)
        demand_requests.append({"dc_id": row.dc_id, "sku_id": row.sku_id, "date": current_date, "mean": mu, "std": std})

    if not demand_requests:
        return pd.DataFrame(columns=["dc_id", "sku_id", "date", "tft_daily_pred", "tft_pred_len"])

    preds = []
    with torch.no_grad():
        for start in range(0, len(demand_requests), batch_size):
            end = start + batch_size
            batch_pred = model(
                torch.tensor(np.array(x_enc_list[start:end]), dtype=torch.float32).to(device),
                torch.tensor(np.array(x_mark_enc_list[start:end]), dtype=torch.float32).to(device),
                torch.tensor(np.array(x_dec_list[start:end]), dtype=torch.float32).to(device),
                torch.tensor(np.array(x_mark_dec_list[start:end]), dtype=torch.float32).to(device),
            ).cpu().numpy()
            preds.append(batch_pred)
    preds = np.concatenate(preds, axis=0)

    rows = []
    for meta, pred in zip(demand_requests, preds):
        median_pred = pred[:, mid_idx]
        denorm_pred = np.maximum(median_pred * meta["std"] + meta["mean"], 0.0)
        rows.append(
            {
                "dc_id": meta["dc_id"],
                "sku_id": meta["sku_id"],
                "date": meta["date"],
                "tft_daily_pred": json.dumps([float(x) for x in denorm_pred]),
                "tft_pred_len": pred_len,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class DecisionArtifacts:
    feature_columns: List[str]
    scaler: StandardScaler


def choose_factory(dc_id: str, tariff: pd.DataFrame) -> Tuple[str, float]:
    sub = tariff[tariff["dc_id"] == dc_id].sort_values("unit_cost")
    if sub.empty:
        return "UNKNOWN", 0.0
    row = sub.iloc[0]
    return str(row["factory_id"]), float(row["unit_cost"])


def round_to_pallet(qty: float, pt_box: float) -> float:
    pt_box = max(float(pt_box), 1.0)
    return math.ceil(max(qty, 0.0) / pt_box) * pt_box


def generate_decision_dataset(
    sales: pd.DataFrame,
    lead: pd.DataFrame,
    dc_inventory: pd.DataFrame,
    factory_inventory: pd.DataFrame,
    dc_capacity: pd.DataFrame,
    push_limit: pd.DataFrame,
    tariff: pd.DataFrame,
    unit_rate: pd.DataFrame,
    vlt_predictions: Optional[pd.DataFrame] = None,
    demand_predictions: Optional[pd.DataFrame] = None,
    demand_history_days: int = 30,
    safety_factor: float = 0.25,
):
    inventory_map = dc_inventory.set_index(["dc_id", "sku_id"])["available_inv"].to_dict()
    capacity_map = dc_capacity.set_index("dc_id")["capacity"].to_dict()
    unit_map = unit_rate.set_index("sku_id")[["pt_box", "box_volume"]].to_dict("index")
    factory_stock = factory_inventory.groupby(["factory_id", "sku_id"])["available_inv"].sum().to_dict()
    push_map = push_limit.set_index("factory_id")[["push_lb", "push_ub"]].to_dict("index")
    if vlt_predictions is not None:
        vlt_pred_map = vlt_predictions.set_index(["dc_id", "sku_id", "date"])["vlt_pred"].to_dict()
    else:
        vlt_pred_map = {}
    if demand_predictions is not None and not demand_predictions.empty:
        demand_pred_map = demand_predictions.set_index(["dc_id", "sku_id", "date"])["tft_daily_pred"].to_dict()
    else:
        demand_pred_map = {}

    rows = []
    merged = lead.merge(sales[["date", "dc_id", "sku_id", "sales_qty"]], on=["date", "dc_id", "sku_id"], how="left")
    sales_group = {k: v.sort_values("date").reset_index(drop=True) for k, v in sales.groupby(["dc_id", "sku_id"])}

    for row in merged.itertuples(index=False):
        key = (row.dc_id, row.sku_id)
        grp = sales_group.get(key)
        if grp is None:
            continue
        current_date = pd.Timestamp(row.date)
        pos = grp.index[grp["date"] == current_date]
        if len(pos) == 0:
            continue
        pos = int(pos[0])
        if pos < demand_history_days:
            continue

        leadtime_actual = float(row.leadtime)
        future_end = pos + int(max(1, round(leadtime_actual)))
        if future_end >= len(grp):
            continue

        history = grp.iloc[pos - demand_history_days:pos]
        future = grp.iloc[pos + 1: future_end + 1]
        if history.empty or future.empty:
            continue

        vlt_pred = float(vlt_pred_map.get((row.dc_id, row.sku_id, current_date), leadtime_actual))
        horizon = max(1, int(round(vlt_pred)))
        demand_pred_raw = demand_pred_map.get((row.dc_id, row.sku_id, current_date))
        if demand_pred_raw is not None:
            demand_pred_seq = np.array(json.loads(demand_pred_raw), dtype=np.float32)
            demand_forecast_sum = sum_forecast_horizon(demand_pred_seq, horizon)
            demand_source = "tft"
        else:
            demand_forecast_sum = rolling_demand_forecast(history["sales_qty"].values.astype(float), horizon)
            demand_source = "rolling_proxy"
        future_demand_sum = float(future["sales_qty"].sum())
        safety_stock = safety_factor * history["sales_qty"].tail(7).mean() * horizon
        current_inv = float(inventory_map.get(key, 0.0))
        dc_capacity_value = float(capacity_map.get(row.dc_id, 0.0))

        unit_info = unit_map.get(row.sku_id, {"pt_box": 1.0, "box_volume": 1.0})
        pt_box = float(unit_info["pt_box"])
        box_volume = float(unit_info["box_volume"])
        factory_id, unit_cost = choose_factory(row.dc_id, tariff)
        factory_inv = float(factory_stock.get((factory_id, row.sku_id), 0.0))
        push_cfg = push_map.get(factory_id, {"push_lb": 0.0, "push_ub": float("inf")})

        unconstrained_target = max(0.0, future_demand_sum + safety_stock - 0.35 * current_inv)
        rounded_target = round_to_pallet(unconstrained_target, pt_box)
        capacity_limit = dc_capacity_value / max(box_volume, 1e-6)
        feasible_target = min(rounded_target, factory_inv, capacity_limit, float(push_cfg["push_ub"]))
        feasible_target = max(0.0, feasible_target)

        rows.append(
            {
                "date": current_date,
                "dc_id": row.dc_id,
                "sku_id": row.sku_id,
                "leadtime_actual": leadtime_actual,
                "vlt_pred": vlt_pred,
                "recent_mean_7": float(history["sales_qty"].tail(7).mean()),
                "recent_mean_30": float(history["sales_qty"].mean()),
                "recent_std_30": float(history["sales_qty"].std(ddof=0)),
                "recent_sum_7": float(history["sales_qty"].tail(7).sum()),
                "recent_sum_30": float(history["sales_qty"].sum()),
                "demand_forecast_sum": demand_forecast_sum,
                "future_demand_sum": future_demand_sum,
                "demand_source": demand_source,
                "current_inv": current_inv,
                "dc_capacity": dc_capacity_value,
                "factory_inv": factory_inv,
                "transport_cost": unit_cost,
                "pt_box": pt_box,
                "box_volume": box_volume,
                "push_lb": float(push_cfg["push_lb"]),
                "push_ub": float(push_cfg["push_ub"]),
                "target_replenishment": feasible_target,
            }
        )

    return pd.DataFrame(rows)


def build_decision_loaders(decision_df: pd.DataFrame, batch_size: int = 256):
    feature_cols = [
        "leadtime_actual",
        "vlt_pred",
        "recent_mean_7",
        "recent_mean_30",
        "recent_std_30",
        "recent_sum_7",
        "recent_sum_30",
        "demand_forecast_sum",
        "current_inv",
        "dc_capacity",
        "factory_inv",
        "transport_cost",
        "pt_box",
        "box_volume",
        "push_lb",
        "push_ub",
    ]
    train_df, val_df, test_df = split_by_date(decision_df, "date")
    scaler = StandardScaler().fit(train_df[feature_cols].values.astype(np.float32))

    def make_loader(df, shuffle=False):
        x = scaler.transform(df[feature_cols].values.astype(np.float32))
        y = df["target_replenishment"].values.astype(np.float32)
        return DataLoader(TabularDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    return (
        make_loader(train_df, shuffle=True),
        make_loader(val_df, shuffle=False),
        make_loader(test_df, shuffle=False),
        DecisionArtifacts(feature_columns=feature_cols, scaler=scaler),
        train_df,
        val_df,
        test_df,
    )


def align_decision_dataset_with_labels(
    decision_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_target_col: str = "optimal_replenishment_box",
    keep_only_order_days: bool = True,
):
    aligned = decision_df.copy()
    labels = labels_df.copy()
    aligned["date"] = pd.to_datetime(aligned["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    merge_cols = ["dc_id", "sku_id", "date"]
    label_cols = merge_cols + [
        "is_order_day",
        "optimal_replenishment_pallet",
        "optimal_replenishment_box",
        "arrival_qty_box",
        "ending_inventory_box",
        "stockout_qty_box",
        "holding_cost",
        "stockout_cost",
        "unit_transport_cost",
    ]
    aligned = aligned.merge(labels[label_cols], on=merge_cols, how="inner")
    if keep_only_order_days:
        aligned = aligned[aligned["is_order_day"] == 1].copy()

    aligned["target_replenishment"] = aligned[label_target_col].astype(np.float32)
    aligned["target_replenishment_pallet"] = aligned["optimal_replenishment_pallet"].astype(np.float32)
    aligned["target_replenishment_box"] = aligned["optimal_replenishment_box"].astype(np.float32)
    aligned = aligned.sort_values(["date", "dc_id", "sku_id"]).reset_index(drop=True)
    return aligned


def train_decision_model_from_dataframe(decision_df: pd.DataFrame, args, device):
    train_loader, val_loader, test_loader, artifacts, train_df, val_df, test_df = build_decision_loaders(
        decision_df,
        batch_size=args.batch_size,
    )
    model = ReplenishmentMLP(
        input_dim=len(artifacts.feature_columns),
        hidden_dim=args.mlp_hidden_size,
        dropout=args.dropout,
    ).to(device)
    history = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)
    _, pred, target = evaluate_epoch(model, test_loader, device)
    metrics = compute_regression_metrics(pred, target)
    return model, artifacts, history, metrics, train_df, val_df, test_df


def train_vlt_module(args, device):
    data_dir = Path(args.data_dir)
    _, lead, *_ = load_reference_tables(data_dir)
    lead["leadtime"] = pd.to_numeric(lead["leadtime"], errors="coerce").fillna(0.0)
    x, y, meta = build_vlt_samples(lead, seq_len=args.vlt_seq_len)

    train_meta, val_meta, test_meta = split_by_date(meta, "date")
    train_idx = meta["date"].isin(train_meta["date"])
    val_idx = meta["date"].isin(val_meta["date"])
    test_idx = meta["date"].isin(test_meta["date"])

    x_scaler = StandardScaler().fit(x[train_idx])
    y_scaler = StandardScaler().fit(y[train_idx].reshape(-1, 1))
    x_scaled = x_scaler.transform(x)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1)

    train_loader = DataLoader(SequenceDataset(x_scaled[train_idx], y_scaled[train_idx]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(x_scaled[val_idx], y_scaled[val_idx]), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(x_scaled[test_idx], y_scaled[test_idx]), batch_size=args.batch_size, shuffle=False)

    model = LeadTimeLSTM(input_size=x.shape[-1], hidden_size=args.hidden_size, num_layers=2, dropout=args.dropout).to(device)
    history = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    _, pred_scaled, target_scaled = evaluate_epoch(model, test_loader, device)
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    target = y_scaler.inverse_transform(target_scaled.reshape(-1, 1)).reshape(-1)
    metrics = compute_regression_metrics(pred, target)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "vlt_lstm.pt")
    torch.save({"x_mean": x_scaler.mean_, "x_std": x_scaler.std_, "y_mean": y_scaler.mean_, "y_std": y_scaler.std_}, out_dir / "vlt_scalers.pt")

    full_loader = DataLoader(SequenceDataset(x_scaled, y_scaled), batch_size=args.batch_size, shuffle=False)
    pred_batches = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in full_loader:
            batch_pred = model(batch_x.to(device)).cpu().numpy()
            pred_batches.append(batch_pred)
    pred_all_scaled = np.concatenate(pred_batches)
    pred_all = y_scaler.inverse_transform(pred_all_scaled.reshape(-1, 1)).reshape(-1)
    pred_frame = meta.copy()
    pred_frame["vlt_pred"] = pred_all
    pred_frame.to_csv(out_dir / "vlt_predictions.csv", index=False)
    with open(out_dir / "vlt_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": history}, f, ensure_ascii=False, indent=2)
    print("VLT metrics:", metrics)
    return pred_frame


def train_decision_module(
    args,
    device,
    vlt_predictions: Optional[pd.DataFrame],
    demand_predictions: Optional[pd.DataFrame],
):
    data_dir = Path(args.data_dir)
    sales, lead, dc_inventory, factory_inventory, dc_capacity, push_limit, tariff, unit_rate = load_reference_tables(data_dir)
    lead["leadtime"] = pd.to_numeric(lead["leadtime"], errors="coerce").fillna(0.0)

    decision_df = generate_decision_dataset(
        sales=sales,
        lead=lead,
        dc_inventory=dc_inventory,
        factory_inventory=factory_inventory,
        dc_capacity=dc_capacity,
        push_limit=push_limit,
        tariff=tariff,
        unit_rate=unit_rate,
        vlt_predictions=vlt_predictions,
        demand_predictions=demand_predictions,
        demand_history_days=args.demand_history_days,
        safety_factor=args.safety_factor,
    )
    if decision_df.empty:
        raise RuntimeError("No decision samples were generated. Please reduce history length or inspect source data.")

    train_loader, val_loader, test_loader, artifacts, train_df, val_df, test_df = build_decision_loaders(decision_df, batch_size=args.batch_size)
    model = ReplenishmentMLP(input_dim=len(artifacts.feature_columns), hidden_dim=args.mlp_hidden_size, dropout=args.dropout).to(device)
    history = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    _, pred, target = evaluate_epoch(model, test_loader, device)
    metrics = compute_regression_metrics(pred, target)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "replenishment_mlp.pt")
    torch.save({"mean": artifacts.scaler.mean_, "std": artifacts.scaler.std_, "feature_columns": artifacts.feature_columns}, out_dir / "decision_scaler.pt")
    decision_df.to_csv(out_dir / "decision_dataset.csv", index=False)
    with open(out_dir / "decision_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": history}, f, ensure_ascii=False, indent=2)
    print("Decision metrics:", metrics)
    return decision_df, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Full TFT-MPIR style pipeline: VLT forecast + replenishment decision")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/full_pipeline")
    parser.add_argument("--tft_checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--tft_batch_size", type=int, default=512)
    parser.add_argument("--tft_test_days", type=int, default=30)
    parser.add_argument("--tft_val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--mlp_hidden_size", type=int, default=128)
    parser.add_argument("--vlt_seq_len", type=int, default=14)
    parser.add_argument("--demand_history_days", type=int, default=30)
    parser.add_argument("--safety_factor", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vlt_predictions = train_vlt_module(args, device)
    sales, lead, *_ = load_reference_tables(Path(args.data_dir))
    demand_predictions = build_tft_demand_prediction_frame(
        sales=sales,
        lead=lead,
        checkpoint_dir=Path(args.tft_checkpoint_dir),
        device=device,
        batch_size=args.tft_batch_size,
        test_days=args.tft_test_days,
        val_ratio=args.tft_val_ratio,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    demand_predictions.to_csv(Path(args.output_dir) / "tft_demand_predictions.csv", index=False)
    train_decision_module(
        args,
        device,
        vlt_predictions=vlt_predictions,
        demand_predictions=demand_predictions,
    )
    print(f"Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
