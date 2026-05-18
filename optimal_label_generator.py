import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


def load_data(data_dir: Path):
    sales = pd.read_csv(data_dir / "sales_data.csv")
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales.rename(columns={"sale": "sales_qty"})

    lead = pd.read_csv(data_dir / "leadtime_data.csv")
    lead["date"] = pd.to_datetime(lead["date"])
    lead["leadtime"] = pd.to_numeric(lead["leadtime"], errors="coerce").fillna(0).astype(int)
    idx_split = lead["idx"].str.split("_", expand=True)
    lead["dc_id"] = idx_split[0]
    lead["sku_id"] = idx_split[1]

    dc_inventory = pd.read_csv(data_dir / "dc_inventory.csv")
    dc_inventory["available_inv"] = pd.to_numeric(dc_inventory["available_inv"], errors="coerce").fillna(0.0)

    tariff = pd.read_csv(data_dir / "transport_tariff.csv")
    tariff["unit_cost"] = pd.to_numeric(tariff["unit_cost"], errors="coerce").fillna(0.0)

    unit_rate = pd.read_csv(data_dir / "unit_rate.csv")
    unit_rate["pt_box"] = pd.to_numeric(unit_rate["pt_box"], errors="coerce").fillna(1.0)

    return sales, lead, dc_inventory, tariff, unit_rate


def choose_cheapest_factory(dc_id: str, tariff: pd.DataFrame):
    sub = tariff[tariff["dc_id"] == dc_id].sort_values("unit_cost")
    if sub.empty:
        return "UNKNOWN", 0.0
    row = sub.iloc[0]
    return str(row["factory_id"]), float(row["unit_cost"])


def simulate_inventory_path(initial_inventory, demand, leadtime, order_qty_pallets, pt_box):
    horizon = len(demand)
    arrivals = np.zeros(horizon, dtype=np.float64)
    for t, qty in enumerate(order_qty_pallets):
        arrival_day = t + int(leadtime[t])
        if 0 <= arrival_day < horizon:
            arrivals[arrival_day] += qty * pt_box

    inventory = np.zeros(horizon, dtype=np.float64)
    stockout = np.zeros(horizon, dtype=np.float64)
    prev_inventory = float(initial_inventory)
    for t in range(horizon):
        net = prev_inventory + arrivals[t] - demand[t]
        inventory[t] = max(net, 0.0)
        stockout[t] = max(-net, 0.0)
        prev_inventory = inventory[t]
    return arrivals, inventory, stockout


def solve_dcsku_milp(
    demand,
    leadtime,
    initial_inventory,
    unit_transport_cost,
    pt_box,
    holding_cost,
    stockout_cost,
    order_mask,
):
    horizon = len(demand)
    n = horizon
    q_offset = 0
    m_offset = n
    s_offset = 2 * n
    n_vars = 3 * n

    c = np.zeros(n_vars, dtype=np.float64)
    c[q_offset:q_offset + n] = pt_box * unit_transport_cost
    c[m_offset:m_offset + n] = holding_cost
    c[s_offset:s_offset + n] = stockout_cost

    ub_q = max(int(np.ceil((initial_inventory + demand.sum()) / max(pt_box, 1.0))) + 5, 1)
    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, np.inf, dtype=np.float64)
    for t in range(n):
        ub[q_offset + t] = ub_q if order_mask[t] else 0.0

    integrality = np.zeros(n_vars, dtype=int)
    integrality[q_offset:q_offset + n] = 1

    aeq = np.zeros((n, n_vars), dtype=np.float64)
    beq = np.zeros(n, dtype=np.float64)

    arrivals_by_day = {}
    for order_day in range(n):
        arrival_day = order_day + int(leadtime[order_day])
        if 0 <= arrival_day < n:
            arrivals_by_day.setdefault(arrival_day, []).append(order_day)

    for t in range(n):
        aeq[t, m_offset + t] = 1.0
        aeq[t, s_offset + t] = -1.0
        if t > 0:
            aeq[t, m_offset + t - 1] = -1.0
            beq[t] = -float(demand[t])
        else:
            beq[t] = float(initial_inventory) - float(demand[t])
        for order_day in arrivals_by_day.get(t, []):
            aeq[t, q_offset + order_day] = -float(pt_box)

    constraints = LinearConstraint(aeq, beq, beq)
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"time_limit": 30},
    )
    if not result.success:
        raise RuntimeError("MILP failed: {}".format(result.message))

    solution = result.x
    q = np.round(solution[q_offset:q_offset + n]).astype(int)
    m = solution[m_offset:m_offset + n]
    s = solution[s_offset:s_offset + n]
    return q, m, s, float(result.fun)


def generate_optimal_labels(
    sales: pd.DataFrame,
    lead: pd.DataFrame,
    dc_inventory: pd.DataFrame,
    tariff: pd.DataFrame,
    unit_rate: pd.DataFrame,
    holding_cost_ratio: float = 0.01,
    stockout_to_transport_ratio: float = 1.5,
    order_weekdays=(0, 4),
    limit_groups: int = 0,
):
    sales_group = {
        key: grp.sort_values("date").reset_index(drop=True)
        for key, grp in sales.groupby(["dc_id", "sku_id"])
    }
    inventory_map = dc_inventory.set_index(["dc_id", "sku_id"])["available_inv"].to_dict()
    pt_box_map = unit_rate.set_index("sku_id")["pt_box"].to_dict()

    all_rows = []
    summary_rows = []
    groups = list(lead.groupby(["dc_id", "sku_id"]))
    if limit_groups > 0:
        groups = groups[:limit_groups]

    for idx, ((dc_id, sku_id), lead_grp) in enumerate(groups, start=1):
        lead_grp = lead_grp.sort_values("date").reset_index(drop=True)
        sales_grp = sales_group.get((dc_id, sku_id))
        if sales_grp is None:
            continue

        horizon_dates = lead_grp["date"].tolist()
        sales_slice = sales_grp[sales_grp["date"].isin(horizon_dates)].sort_values("date").reset_index(drop=True)
        if len(sales_slice) != len(lead_grp):
            continue

        demand = sales_slice["sales_qty"].astype(float).values
        leadtime = lead_grp["leadtime"].astype(int).values
        initial_inventory = float(inventory_map.get((dc_id, sku_id), 0.0))
        pt_box = float(pt_box_map.get(sku_id, 1.0))
        factory_id, unit_transport_cost = choose_cheapest_factory(dc_id, tariff)
        stockout_cost = stockout_to_transport_ratio * unit_transport_cost
        holding_cost = holding_cost_ratio * stockout_cost
        order_mask = lead_grp["date"].dt.weekday.isin(order_weekdays).values.astype(bool)

        q, m, s, objective = solve_dcsku_milp(
            demand=demand,
            leadtime=leadtime,
            initial_inventory=initial_inventory,
            unit_transport_cost=unit_transport_cost,
            pt_box=pt_box,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            order_mask=order_mask,
        )
        arrivals, sim_inventory, sim_stockout = simulate_inventory_path(
            initial_inventory=initial_inventory,
            demand=demand,
            leadtime=leadtime,
            order_qty_pallets=q,
            pt_box=pt_box,
        )

        transport_cost = float((q * pt_box * unit_transport_cost).sum())
        holding_cost_total = float(sim_inventory.sum() * holding_cost)
        stockout_cost_total = float(sim_stockout.sum() * stockout_cost)

        summary_rows.append(
            {
                "dc_id": dc_id,
                "sku_id": sku_id,
                "factory_id": factory_id,
                "unit_transport_cost": unit_transport_cost,
                "holding_cost": holding_cost,
                "stockout_cost": stockout_cost,
                "objective": objective,
                "transport_cost": transport_cost,
                "holding_cost_total": holding_cost_total,
                "stockout_cost_total": stockout_cost_total,
                "total_cost_check": transport_cost + holding_cost_total + stockout_cost_total,
            }
        )

        for t, day in enumerate(horizon_dates):
            all_rows.append(
                {
                    "dc_id": dc_id,
                    "sku_id": sku_id,
                    "date": day,
                    "leadtime": int(leadtime[t]),
                    "sales_qty": float(demand[t]),
                    "is_order_day": int(order_mask[t]),
                    "initial_inventory": initial_inventory if t == 0 else float(sim_inventory[t - 1]),
                    "arrival_qty_box": float(arrivals[t]),
                    "ending_inventory_box": float(sim_inventory[t]),
                    "stockout_qty_box": float(sim_stockout[t]),
                    "optimal_replenishment_pallet": int(q[t]),
                    "optimal_replenishment_box": float(q[t] * pt_box),
                    "pt_box": pt_box,
                    "factory_id": factory_id,
                    "unit_transport_cost": unit_transport_cost,
                    "holding_cost": holding_cost,
                    "stockout_cost": stockout_cost,
                }
            )

        if idx % 100 == 0:
            print("Solved {} / {} DC-SKU groups".format(idx, len(groups)))

    labels_df = pd.DataFrame(all_rows).sort_values(["dc_id", "sku_id", "date"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["dc_id", "sku_id"]).reset_index(drop=True)
    return labels_df, summary_df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate post-hoc optimal replenishment labels with MILP.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/optimal_labels")
    parser.add_argument("--holding_cost_ratio", type=float, default=0.01)
    parser.add_argument("--stockout_to_transport_ratio", type=float, default=1.5)
    parser.add_argument("--order_weekdays", type=str, default="0,4")
    parser.add_argument("--limit_groups", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    order_weekdays = tuple(int(x) for x in args.order_weekdays.split(",") if x.strip())
    sales, lead, dc_inventory, tariff, unit_rate = load_data(Path(args.data_dir))
    labels_df, summary_df = generate_optimal_labels(
        sales=sales,
        lead=lead,
        dc_inventory=dc_inventory,
        tariff=tariff,
        unit_rate=unit_rate,
        holding_cost_ratio=args.holding_cost_ratio,
        stockout_to_transport_ratio=args.stockout_to_transport_ratio,
        order_weekdays=order_weekdays,
        limit_groups=args.limit_groups,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_dir / "optimal_replenishment_labels.csv", index=False)
    summary_df.to_csv(output_dir / "optimal_replenishment_summary.csv", index=False)

    metadata = {
        "holding_cost_ratio": args.holding_cost_ratio,
        "stockout_to_transport_ratio": args.stockout_to_transport_ratio,
        "order_weekdays": list(order_weekdays),
        "n_rows": int(len(labels_df)),
        "n_groups": int(summary_df.shape[0]),
        "total_cost_sum": float(summary_df["objective"].sum()) if not summary_df.empty else 0.0,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved labels to", output_dir)
    print(metadata)


if __name__ == "__main__":
    main()
