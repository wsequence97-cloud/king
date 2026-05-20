import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_frames(eval_dir: Path, split_csv: Path):
    sim = pd.read_csv(eval_dir / "decision_cost_simulation.csv")
    groups = pd.read_csv(eval_dir / "decision_group_costs.csv")
    split_df = pd.read_csv(split_csv)
    for df in (sim, split_df):
        df["date"] = pd.to_datetime(df["date"])
    return sim, groups, split_df


def build_group_diagnostics(sim: pd.DataFrame, groups: pd.DataFrame, split_df: pd.DataFrame):
    feature_cols = [
        "demand_forecast_sum",
        "future_demand_sum",
        "vlt_pred",
        "leadtime_actual",
        "current_inv",
        "transport_cost",
        "factory_inv",
        "dc_capacity",
    ]
    merge_cols = ["date", "dc_id", "sku_id"] + [c for c in feature_cols if c in split_df.columns and c not in sim.columns]
    merged = sim.merge(split_df[merge_cols], on=["date", "dc_id", "sku_id"], how="left")

    diag = (
        merged.groupby(["dc_id", "sku_id"], as_index=False)
        .agg(
            n_days=("date", "count"),
            n_order_days=("is_order_day", "sum"),
            n_pred_stockout_days=("pred_stockout_box", lambda s: int((s > 0).sum())),
            n_opt_stockout_days=("stockout_qty_box", lambda s: int((s > 0).sum())),
            pred_replenishment_sum=("pred_replenishment_box", "sum"),
            opt_replenishment_sum=("optimal_replenishment_box", "sum"),
            pred_replenishment_mean=("pred_replenishment_box", "mean"),
            opt_replenishment_mean=("optimal_replenishment_box", "mean"),
            pred_stockout_box_sum=("pred_stockout_box", "sum"),
            opt_stockout_box_sum=("stockout_qty_box", "sum"),
            pred_inventory_mean=("pred_ending_inventory_box", "mean"),
            opt_inventory_mean=("ending_inventory_box", "mean"),
            sales_sum=("sales_qty", "sum"),
            sales_mean=("sales_qty", "mean"),
            demand_forecast_sum_mean=("demand_forecast_sum", "mean"),
            future_demand_sum_mean=("future_demand_sum", "mean"),
            vlt_pred_mean=("vlt_pred", "mean"),
            leadtime_mean=("leadtime", "mean"),
            current_inv_mean=("current_inv", "mean"),
            transport_cost_mean=("transport_cost", "mean"),
            pt_box_mean=("pt_box", "mean"),
        )
    )
    out = groups.merge(diag, on=["dc_id", "sku_id"], how="left")
    out["stockout_gap"] = out["pred_stockout_cost"] - out["opt_stockout_cost"]
    out["holding_gap"] = out["pred_holding_cost"] - out["opt_holding_cost"]
    out["transport_gap"] = out["pred_transport_cost"] - out["opt_transport_cost"]
    out["total_gap"] = out["pred_total_cost"] - out["opt_total_cost"]
    out["replenishment_gap"] = out["pred_replenishment_sum"] - out["opt_replenishment_sum"]
    out["forecast_bias"] = out["demand_forecast_sum_mean"] - out["future_demand_sum_mean"]
    out["pred_stockout_day_ratio"] = np.where(out["n_days"] > 0, out["n_pred_stockout_days"] / out["n_days"], np.nan)
    out["opt_stockout_day_ratio"] = np.where(out["n_days"] > 0, out["n_opt_stockout_days"] / out["n_days"], np.nan)
    out = out.sort_values(["pred_stockout_cost", "total_gap"], ascending=[False, False]).reset_index(drop=True)
    return merged, out


def build_topk_comparison(groups_diag: pd.DataFrame, ks):
    base = {
        "pred_transport_cost": float(groups_diag["pred_transport_cost"].sum()),
        "pred_holding_cost": float(groups_diag["pred_holding_cost"].sum()),
        "pred_stockout_cost": float(groups_diag["pred_stockout_cost"].sum()),
        "pred_total_cost": float(groups_diag["pred_total_cost"].sum()),
        "opt_transport_cost": float(groups_diag["opt_transport_cost"].sum()),
        "opt_holding_cost": float(groups_diag["opt_holding_cost"].sum()),
        "opt_stockout_cost": float(groups_diag["opt_stockout_cost"].sum()),
        "opt_total_cost": float(groups_diag["opt_total_cost"].sum()),
    }
    rows = []
    ranked = groups_diag.sort_values(["pred_stockout_cost", "total_gap"], ascending=[False, False]).reset_index(drop=True)
    for k in ks:
        excluded = ranked.head(k)
        kept = ranked.iloc[k:]
        if kept.empty:
            continue
        row = {
            "top_k_excluded": int(k),
            "excluded_group_count": int(len(excluded)),
            "kept_group_count": int(len(kept)),
            "excluded_pred_stockout_cost": float(excluded["pred_stockout_cost"].sum()),
            "excluded_total_gap": float(excluded["total_gap"].sum()),
            "pred_transport_cost_kept": float(kept["pred_transport_cost"].sum()),
            "pred_holding_cost_kept": float(kept["pred_holding_cost"].sum()),
            "pred_stockout_cost_kept": float(kept["pred_stockout_cost"].sum()),
            "pred_total_cost_kept": float(kept["pred_total_cost"].sum()),
            "opt_transport_cost_kept": float(kept["opt_transport_cost"].sum()),
            "opt_holding_cost_kept": float(kept["opt_holding_cost"].sum()),
            "opt_stockout_cost_kept": float(kept["opt_stockout_cost"].sum()),
            "opt_total_cost_kept": float(kept["opt_total_cost"].sum()),
            "pred_total_cost_reduction_vs_all_pct": (base["pred_total_cost"] - float(kept["pred_total_cost"].sum())) / base["pred_total_cost"] * 100.0,
        }
        opt_total = row["opt_total_cost_kept"]
        row["total_gap_pct_vs_opt_kept"] = ((row["pred_total_cost_kept"] - opt_total) / opt_total * 100.0) if opt_total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_scenario_summary(merged: pd.DataFrame):
    merged = merged.copy()
    merged["has_pred_stockout"] = merged["pred_stockout_cost"] > 0
    cols = [
        "demand_forecast_sum",
        "future_demand_sum",
        "vlt_pred",
        "leadtime",
        "current_inv",
        "pred_replenishment_box",
        "optimal_replenishment_box",
        "pred_stockout_box",
        "stockout_qty_box",
    ]
    available = [c for c in cols if c in merged.columns]
    summary = merged.groupby("has_pred_stockout")[available].mean().reset_index()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stockout-driver diagnostics and exclusion comparison tables.")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k_list", type=str, default="1,3,5,10,20,50,100")
    parser.add_argument("--detail_top_n", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim, groups, split_df = load_frames(eval_dir, Path(args.split_csv))
    merged, group_diag = build_group_diagnostics(sim, groups, split_df)
    ks = [int(x) for x in args.top_k_list.split(",") if x.strip()]
    topk_df = build_topk_comparison(group_diag, ks)
    scenario_df = build_scenario_summary(merged)
    top_groups = group_diag[["dc_id", "sku_id", "pred_stockout_cost", "total_gap"]].head(args.detail_top_n).copy()
    top_groups["group_rank"] = np.arange(1, len(top_groups) + 1)
    top_groups = top_groups.rename(
        columns={
            "pred_stockout_cost": "group_pred_stockout_cost",
            "total_gap": "group_total_gap",
        }
    )
    top_groups_detail = merged.merge(
        top_groups,
        on=["dc_id", "sku_id"],
        how="inner",
    ).sort_values(["group_rank", "date"], ascending=[True, True])

    group_diag.to_csv(output_dir / "group_diagnostics.csv", index=False)
    topk_df.to_csv(output_dir / "topk_exclusion_comparison.csv", index=False)
    scenario_df.to_csv(output_dir / "stockout_scenario_summary.csv", index=False)
    top_groups_detail.to_csv(output_dir / "top_groups_daily_detail.csv", index=False)

    summary = {
        "n_groups": int(len(group_diag)),
        "n_daily_rows": int(len(merged)),
        "top_group_by_stockout": group_diag.loc[0, ["dc_id", "sku_id"]].to_dict() if not group_diag.empty else {},
        "top_group_pred_stockout_cost": float(group_diag.loc[0, "pred_stockout_cost"]) if not group_diag.empty else 0.0,
    }
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
