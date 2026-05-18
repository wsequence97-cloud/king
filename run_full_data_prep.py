import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from full_pipeline import (
    align_decision_dataset_with_labels,
    build_tft_demand_prediction_frame,
    generate_decision_dataset,
    load_reference_tables,
    split_by_date,
    train_decision_model_from_dataframe,
    train_vlt_module,
)
from optimal_label_generator import generate_optimal_labels, load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-click full label generation and training-set alignment for TFT-MPIR style pipeline."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/full_data_prep")
    parser.add_argument("--tft_checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tft_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--mlp_hidden_size", type=int, default=128)
    parser.add_argument("--vlt_seq_len", type=int, default=14)
    parser.add_argument("--demand_history_days", type=int, default=30)
    parser.add_argument("--safety_factor", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holding_cost_ratio", type=float, default=0.01)
    parser.add_argument("--stockout_to_transport_ratio", type=float, default=1.5)
    parser.add_argument("--order_weekdays", type=str, default="0,4")
    parser.add_argument("--label_target_col", type=str, default="optimal_replenishment_box")
    parser.add_argument("--keep_only_order_days", action="store_true")
    parser.add_argument("--skip_vlt", action="store_true")
    parser.add_argument("--skip_tft", action="store_true")
    parser.add_argument("--train_decision", action="store_true")
    parser.add_argument("--limit_groups", type=int, default=0)
    parser.add_argument("--tft_test_days", type=int, default=30)
    parser.add_argument("--tft_val_ratio", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sales_opt, lead_opt, dc_inventory_opt, tariff_opt, unit_rate_opt = load_data(Path(args.data_dir))
    order_weekdays = tuple(int(x) for x in args.order_weekdays.split(",") if x.strip())
    labels_df, summary_df = generate_optimal_labels(
        sales=sales_opt,
        lead=lead_opt,
        dc_inventory=dc_inventory_opt,
        tariff=tariff_opt,
        unit_rate=unit_rate_opt,
        holding_cost_ratio=args.holding_cost_ratio,
        stockout_to_transport_ratio=args.stockout_to_transport_ratio,
        order_weekdays=order_weekdays,
        limit_groups=args.limit_groups,
    )
    labels_df.to_csv(output_dir / "optimal_replenishment_labels.csv", index=False)
    summary_df.to_csv(output_dir / "optimal_replenishment_summary.csv", index=False)
    print("Saved optimal labels:", len(labels_df), "rows")

    sales, lead, dc_inventory, factory_inventory, dc_capacity, push_limit, tariff, unit_rate = load_reference_tables(Path(args.data_dir))
    lead["leadtime"] = pd.to_numeric(lead["leadtime"], errors="coerce").fillna(0.0)

    vlt_predictions = None
    if not args.skip_vlt:
        vlt_predictions = train_vlt_module(args, device)
        vlt_predictions.to_csv(output_dir / "vlt_predictions.csv", index=False)
    else:
        existing = output_dir / "vlt_predictions.csv"
        if existing.exists():
            vlt_predictions = pd.read_csv(existing)
            vlt_predictions["date"] = pd.to_datetime(vlt_predictions["date"])

    demand_predictions = None
    if not args.skip_tft:
        demand_predictions = build_tft_demand_prediction_frame(
            sales=sales,
            lead=lead,
            checkpoint_dir=Path(args.tft_checkpoint_dir),
            device=device,
            batch_size=args.tft_batch_size,
            test_days=args.tft_test_days,
            val_ratio=args.tft_val_ratio,
        )
        demand_predictions.to_csv(output_dir / "tft_demand_predictions.csv", index=False)
    else:
        existing = output_dir / "tft_demand_predictions.csv"
        if existing.exists():
            demand_predictions = pd.read_csv(existing)
            demand_predictions["date"] = pd.to_datetime(demand_predictions["date"])

    proxy_decision_df = generate_decision_dataset(
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
    proxy_decision_df.to_csv(output_dir / "decision_features_proxy.csv", index=False)

    aligned_df = align_decision_dataset_with_labels(
        decision_df=proxy_decision_df,
        labels_df=labels_df,
        label_target_col=args.label_target_col,
        keep_only_order_days=args.keep_only_order_days,
    )
    aligned_df.to_csv(output_dir / "decision_train_aligned.csv", index=False)

    train_df, val_df, test_df = split_by_date(aligned_df, "date")
    train_df.to_csv(output_dir / "decision_train_split.csv", index=False)
    val_df.to_csv(output_dir / "decision_val_split.csv", index=False)
    test_df.to_csv(output_dir / "decision_test_split.csv", index=False)

    metadata = {
        "device": str(device),
        "labels_rows": int(len(labels_df)),
        "labels_groups": int(summary_df.shape[0]),
        "proxy_feature_rows": int(len(proxy_decision_df)),
        "aligned_rows": int(len(aligned_df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "label_target_col": args.label_target_col,
        "keep_only_order_days": bool(args.keep_only_order_days),
        "used_vlt_predictions": vlt_predictions is not None,
        "used_tft_predictions": demand_predictions is not None,
    }

    if args.train_decision:
        model, artifacts, history, metrics, train_used, val_used, test_used = train_decision_model_from_dataframe(
            aligned_df,
            args,
            device,
        )
        torch.save(model.state_dict(), output_dir / "replenishment_mlp_aligned.pt")
        torch.save(
            {
                "mean": artifacts.scaler.mean_,
                "std": artifacts.scaler.std_,
                "feature_columns": artifacts.feature_columns,
            },
            output_dir / "decision_scaler_aligned.pt",
        )
        with open(output_dir / "decision_aligned_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "history": history}, f, ensure_ascii=False, indent=2)
        metadata["decision_metrics"] = metrics
        metadata["decision_train_rows_used"] = int(len(train_used))
        metadata["decision_val_rows_used"] = int(len(val_used))
        metadata["decision_test_rows_used"] = int(len(test_used))

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved full prep artifacts to", output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
