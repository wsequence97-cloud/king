import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from full_pipeline import ReplenishmentMLP, compute_regression_metrics


def load_scaler_bundle(path: Path):
    bundle = torch.load(path, map_location="cpu")
    mean = np.array(bundle["mean"], dtype=np.float32)
    std = np.array(bundle["std"], dtype=np.float32)
    feature_columns = list(bundle["feature_columns"])
    return mean, std, feature_columns


def build_model(model_path: Path, scaler_path: Path, hidden_dim: int, dropout: float, device: torch.device):
    mean, std, feature_columns = load_scaler_bundle(scaler_path)
    model = ReplenishmentMLP(input_dim=len(feature_columns), hidden_dim=hidden_dim, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, mean, std, feature_columns


def predict_dataframe(df: pd.DataFrame, model, mean, std, feature_columns, device: torch.device):
    x = df[feature_columns].values.astype(np.float32)
    x = (x - mean) / std
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32).to(device)).cpu().numpy()
    out = df.copy()
    out["pred_replenishment_raw_box"] = pred.astype(np.float32)
    out["pred_replenishment_raw_box"] = out["pred_replenishment_raw_box"].clip(lower=0.0)
    return out


def nearest_pallet_boxes(pred_box: float, pt_box: float) -> float:
    pt_box = max(float(pt_box), 1.0)
    pallets = max(0, int(round(float(pred_box) / pt_box)))
    return float(pallets * pt_box)


def simulate_group(group_daily: pd.DataFrame, predicted_orders: dict):
    grp = group_daily.sort_values("date").reset_index(drop=True).copy()
    horizon = len(grp)
    arrivals = np.zeros(horizon, dtype=np.float64)
    inventory = np.zeros(horizon, dtype=np.float64)
    stockout = np.zeros(horizon, dtype=np.float64)
    pred_box = np.zeros(horizon, dtype=np.float64)

    for t in range(horizon):
        row = grp.iloc[t]
        key = (str(row["dc_id"]), str(row["sku_id"]), pd.Timestamp(row["date"]))
        pt_box = float(row["pt_box"])
        pred_box[t] = float(predicted_orders.get(key, 0.0))
        pred_box[t] = nearest_pallet_boxes(pred_box[t], pt_box)
        arrival_day = t + int(row["leadtime"])
        if 0 <= arrival_day < horizon:
            arrivals[arrival_day] += pred_box[t]

    prev_inventory = float(grp.loc[0, "initial_inventory"])
    for t in range(horizon):
        row = grp.iloc[t]
        net = prev_inventory + arrivals[t] - float(row["sales_qty"])
        inventory[t] = max(net, 0.0)
        stockout[t] = max(-net, 0.0)
        prev_inventory = inventory[t]

    grp["pred_replenishment_box"] = pred_box
    grp["pred_arrival_box"] = arrivals
    grp["pred_ending_inventory_box"] = inventory
    grp["pred_stockout_box"] = stockout
    grp["pred_transport_cost"] = grp["pred_replenishment_box"] * grp["unit_transport_cost"]
    grp["pred_holding_cost"] = grp["pred_ending_inventory_box"] * grp["holding_cost"]
    grp["pred_stockout_cost"] = grp["pred_stockout_box"] * grp["stockout_cost"]
    grp["pred_total_cost"] = grp["pred_transport_cost"] + grp["pred_holding_cost"] + grp["pred_stockout_cost"]

    grp["opt_transport_cost"] = grp["optimal_replenishment_box"] * grp["unit_transport_cost"]
    grp["opt_holding_cost"] = grp["ending_inventory_box"] * grp["holding_cost"]
    grp["opt_stockout_cost"] = grp["stockout_qty_box"] * grp["stockout_cost"]
    grp["opt_total_cost"] = grp["opt_transport_cost"] + grp["opt_holding_cost"] + grp["opt_stockout_cost"]
    return grp


def evaluate_costs(split_df: pd.DataFrame, labels_df: pd.DataFrame):
    pred_map = {
        (str(r.dc_id), str(r.sku_id), pd.Timestamp(r.date)): float(r.pred_replenishment_raw_box)
        for r in split_df.itertuples(index=False)
    }
    start_dates = split_df.groupby(["dc_id", "sku_id"])["date"].min().to_dict()

    simulated_groups = []
    for (dc_id, sku_id), start_date in start_dates.items():
        mask = (
            (labels_df["dc_id"] == dc_id)
            & (labels_df["sku_id"] == sku_id)
            & (pd.to_datetime(labels_df["date"]) >= pd.Timestamp(start_date))
        )
        group_daily = labels_df.loc[mask].copy()
        if group_daily.empty:
            continue
        simulated_groups.append(simulate_group(group_daily, pred_map))

    if not simulated_groups:
        return pd.DataFrame(), {}, pd.DataFrame()

    sim_df = pd.concat(simulated_groups, ignore_index=True)
    group_metrics = (
        sim_df.groupby(["dc_id", "sku_id"], as_index=False)
        .agg(
            pred_transport_cost=("pred_transport_cost", "sum"),
            pred_holding_cost=("pred_holding_cost", "sum"),
            pred_stockout_cost=("pred_stockout_cost", "sum"),
            pred_total_cost=("pred_total_cost", "sum"),
            opt_transport_cost=("opt_transport_cost", "sum"),
            opt_holding_cost=("opt_holding_cost", "sum"),
            opt_stockout_cost=("opt_stockout_cost", "sum"),
            opt_total_cost=("opt_total_cost", "sum"),
        )
    )
    group_metrics["total_gap_pct"] = np.where(
        group_metrics["opt_total_cost"] > 0,
        (group_metrics["pred_total_cost"] - group_metrics["opt_total_cost"]) / group_metrics["opt_total_cost"] * 100.0,
        np.nan,
    )

    summary = {
        "pred_transport_cost": float(sim_df["pred_transport_cost"].sum()),
        "pred_holding_cost": float(sim_df["pred_holding_cost"].sum()),
        "pred_stockout_cost": float(sim_df["pred_stockout_cost"].sum()),
        "pred_total_cost": float(sim_df["pred_total_cost"].sum()),
        "opt_transport_cost": float(sim_df["opt_transport_cost"].sum()),
        "opt_holding_cost": float(sim_df["opt_holding_cost"].sum()),
        "opt_stockout_cost": float(sim_df["opt_stockout_cost"].sum()),
        "opt_total_cost": float(sim_df["opt_total_cost"].sum()),
    }
    if summary["opt_total_cost"] > 0:
        summary["total_gap_pct_vs_opt"] = (summary["pred_total_cost"] - summary["opt_total_cost"]) / summary["opt_total_cost"] * 100.0
    else:
        summary["total_gap_pct_vs_opt"] = None
    return sim_df, summary, group_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate replenishment decision model with regression and inventory cost metrics.")
    parser.add_argument("--aligned_csv", type=str, required=True)
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scaler_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aligned_df = pd.read_csv(args.aligned_csv)
    split_df = pd.read_csv(args.split_csv)
    labels_df = pd.read_csv(args.labels_csv)
    aligned_df["date"] = pd.to_datetime(aligned_df["date"])
    split_df["date"] = pd.to_datetime(split_df["date"])
    labels_df["date"] = pd.to_datetime(labels_df["date"])

    model, mean, std, feature_columns = build_model(
        model_path=Path(args.model_path),
        scaler_path=Path(args.scaler_path),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        device=device,
    )

    split_pred_df = predict_dataframe(split_df, model, mean, std, feature_columns, device)
    regression = compute_regression_metrics(
        split_pred_df["pred_replenishment_raw_box"].values.astype(np.float32),
        split_pred_df["target_replenishment"].values.astype(np.float32),
    )

    sim_df, cost_summary, group_metrics = evaluate_costs(split_pred_df, labels_df)
    split_pred_df.to_csv(output_dir / "decision_predictions.csv", index=False)
    sim_df.to_csv(output_dir / "decision_cost_simulation.csv", index=False)
    group_metrics.to_csv(output_dir / "decision_group_costs.csv", index=False)

    result = {
        "device": str(device),
        "n_split_rows": int(len(split_pred_df)),
        "n_sim_rows": int(len(sim_df)),
        "regression_metrics": regression,
        "cost_summary": cost_summary,
    }
    with open(output_dir / "decision_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
