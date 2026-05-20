# TFT-MPIR 复现项目说明

本项目用于复现论文 **TFT-MPIR: An end-to-end multi-period inventory replenishment strategy based on temporal fusion transformer**。

当前项目已经不是单一的 `TFT demand forecast` 实验，而是一条较完整的复现链路，覆盖：

- `Demand Forecast`：基于 `TFT` 的销量预测
- `VLT Forecast`：基于 `LSTM` 的提前期预测
- `Optimal Labeling`：基于 `MILP` 的最优补货标签生成
- `Replenishment Decision`：基于 `MLP` 的补货决策训练
- `Evaluation`：回归误差、库存成本、缺货场景分析

---

## 1. 项目整体结构

项目根目录主要可以分为 5 类内容：

1. 核心训练与推理代码
2. 原始数据和数据说明
3. 论文与辅助分析材料
4. 训练/推理/评估产物
5. 交接与项目状态文档

---

## 2. 每个代码文件在做什么

### 2.1 需求预测相关

- [train.py](D:/AA_postgraduate/thesis/code/my_project/train.py)
  需求预测主训练脚本。读取 `sales_data.csv`，训练 `TFT` 模型，保存：
  - `best_model.pt`
  - `last_checkpoint.pt`
  - `training_curve.png`

- [tft_model.py](D:/AA_postgraduate/thesis/code/my_project/tft_model.py)
  `TFT` 模型定义。

- [data_preprocessing.py](D:/AA_postgraduate/thesis/code/my_project/data_preprocessing.py)
  需求预测数据预处理，将原始销售表转换为 `DC-SKU` 粒度的时序窗口样本。

- [dataset.py](D:/AA_postgraduate/thesis/code/my_project/dataset.py)
  将预处理后的样本包装为 PyTorch `Dataset / DataLoader`。

- [visualize.py](D:/AA_postgraduate/thesis/code/my_project/visualize.py)
  训练过程中的曲线可视化与保存。

### 2.2 补货完整流水线

- [full_pipeline.py](D:/AA_postgraduate/thesis/code/my_project/full_pipeline.py)
  当前最重要的主流程代码，负责：
  - 训练 `VLT LSTM`
  - 读取 `TFT checkpoint`
  - 生成 `TFT demand forecast`
  - 构造补货决策特征
  - 对齐标签
  - 训练补货决策 `MLP`

### 2.3 最优标签生成

- [optimal_label_generator.py](D:/AA_postgraduate/thesis/code/my_project/optimal_label_generator.py)
  基于论文里的 `post-hoc optimal solution` 思路生成最优补货标签。

  核心做法：
  - 读取真实历史销量、真实 lead time、初始库存、运输成本、托盘换算
  - 为每个 `DC-SKU` 建立一个 `MILP`
  - 求出每一天的最优补货量 `q_t*`
  - 输出 `optimal_replenishment_labels.csv`

### 2.4 一键数据准备

- [run_full_data_prep.py](D:/AA_postgraduate/thesis/code/my_project/run_full_data_prep.py)
  一键式数据准备和训练脚本，适合上云使用。

  能做的事情包括：
  - 直接生成最优标签，或复用已有标签
  - 训练/生成 `VLT` 预测
  - 生成 `TFT demand` 预测
  - 构造补货特征表
  - 对齐最优标签
  - 切分 `train / val / test`
  - 可选直接训练补货决策模型

### 2.5 补货决策评估与误差分析

- [evaluate_replenishment_decision.py](D:/AA_postgraduate/thesis/code/my_project/evaluate_replenishment_decision.py)
  对训练好的补货决策模型做评估，输出两类指标：
  - 回归误差：`MAE / RMSE / MAPE`
  - 业务成本：运输、持有、缺货、总成本，并与 `OPT` 对比

- [analyze_stockout_drivers.py](D:/AA_postgraduate/thesis/code/my_project/analyze_stockout_drivers.py)
  基于评估结果做缺货驱动分析，输出：
  - `DC-SKU` 级诊断表
  - 剔除 `Top-k` 高缺货组后的结果对比
  - 高缺货组的日级明细

---

## 3. 每个文件夹里是什么

### 3.1 `data/`

[data](D:/AA_postgraduate/thesis/code/my_project/data) 是原始数据目录。

里面的文件含义如下：

- `sales_data.csv`
  历史销量数据。
  是 `TFT demand forecast` 的原始输入，也是后续补货问题里的真实需求。

- `leadtime_data.csv`
  历史提前期数据。
  是 `VLT forecast` 的原始输入，也是最优标签问题里的真实 lead time。

- `dc_inventory.csv`
  `DC-SKU` 的初始库存。

- `factory_inventory.csv`
  工厂侧各 SKU 的可用库存。

- `dc_capacity.csv`
  每个 `DC` 的库容限制。

- `push_limit.csv`
  工厂发货上下限。

- `transport_tariff.csv`
  工厂到 DC 的单位运输成本。

- `unit_rate.csv`
  SKU 的托盘与箱数换算关系，以及体积信息。

### 3.2 `checkpoints/`

[checkpoints](D:/AA_postgraduate/thesis/code/my_project/checkpoints) 是所有训练、推理、评估产物的目录。

#### 根目录下的重要文件

- `best_model.pt`
  当前 demand forecast 最优模型权重。

- `last_checkpoint.pt`
  当前 demand forecast 完整 checkpoint，包含 `model_cfg`。

- `training_curve.png`
  demand forecast 训练曲线。

#### 子目录含义

- `optimal_labels_full/`
  全量 `MILP` 最优补货标签。

  里面关键文件：
  - `optimal_replenishment_labels.csv`
  - `optimal_replenishment_summary.csv`
  - `failed_groups.csv`
  - `metadata.json`

- `full_data_prep_full/`
  基于已有 labels 生成的对齐训练集，使用的是代理特征版本。

- `full_data_prep_full_pred/`
  当前最重要的正式版本。
  使用的是：
  - 真实 `TFT demand predictions`
  - 真实 `VLT predictions`
  - 全量 `optimal labels`

  里面关键文件：
  - `tft_demand_predictions.csv`
  - `vlt_predictions.csv`
  - `decision_train_aligned.csv`
  - `decision_train_split.csv`
  - `decision_val_split.csv`
  - `decision_test_split.csv`
  - `replenishment_mlp_aligned.pt`
  - `decision_scaler_aligned.pt`
  - `decision_aligned_metrics.json`

- `full_data_prep_full_pred/decision_eval/`
  补货决策模型最终评估结果。

- `full_data_prep_full_pred/decision_eval/stockout_analysis/`
  缺货成本分析结果。

- `*_smoke/`
  各类烟测/小规模测试目录，用来验证流程是否跑通。

### 3.3 其他辅助目录

- `.vscode/`
  VS Code 配置。

- `__pycache__/`
  Python 缓存文件。

- `.ipynb_checkpoints/`
  Notebook 自动保存检查点。

---

## 4. 论文与辅助材料

- [Guo 等 - 2025 - TFT-MPIR An end-to-end multi-period inventory replenishment strategy based on temporal fusion trans.pdf](D:/AA_postgraduate/thesis/code/my_project/Guo%20等%20-%202025%20-%20TFT-MPIR%20An%20end-to-end%20multi-period%20inventory%20replenishment%20strategy%20based%20on%20temporal%20fusion%20trans.pdf)
  复现目标论文。

- [data_exp.md](D:/AA_postgraduate/thesis/code/my_project/data_exp.md)
  对公开数据内容的文字整理。

- `image.png` 到 `image-7.png`
  数据字段说明截图。

- [paper_extract_p3_p5.txt](D:/AA_postgraduate/thesis/code/my_project/paper_extract_p3_p5.txt)
- [paper_extract_p7_p11.txt](D:/AA_postgraduate/thesis/code/my_project/paper_extract_p7_p11.txt)
  从论文 PDF 中抽取的文本片段，用于本地分析目标函数、约束、实验设定。

---

## 5. 当前项目已经做到哪一步

当前项目已经完成：

1. `TFT demand forecast` 训练与推理
2. `VLT LSTM` 训练与推理
3. 全量最优补货标签生成
4. 全量训练集对齐
5. 补货决策模型训练
6. 最终成本评估
7. 高缺货成本组分析

也就是说：

**项目已经不是“搭框架阶段”，而是“流程已完整跑通，但最终效果还不理想”的阶段。**

当前最关键的实验结论是：

- 代码链路已经跑通
- 最终补货决策模型效果与 `OPT` 仍有明显差距
- 主要问题集中在缺货成本过高

---

## 6. 推荐运行顺序

### 第一步：训练或确认 demand forecast 模型

如果下面这两个文件已经存在，就不用重复训练：
- `checkpoints/best_model.pt`
- `checkpoints/last_checkpoint.pt`

否则运行：

```bash
python train.py --data data/sales_data.csv
```

### 第二步：生成全量最优标签

```bash
python optimal_label_generator.py --output_dir checkpoints/optimal_labels_full --resume --time_limit 60
```

说明：
- `MILP` 主要吃 CPU
- `--resume` 用于断点续跑
- `--time_limit 60` 是当前已验证过的稳定设置

### 第三步：基于已有全量 labels 生成正式对齐集

```bash
python run_full_data_prep.py --output_dir checkpoints/full_data_prep_full_pred --existing_labels_csv checkpoints/optimal_labels_full/optimal_replenishment_labels.csv --existing_labels_summary_csv checkpoints/optimal_labels_full/optimal_replenishment_summary.csv --keep_only_order_days
```

这一步会：
- 复用已有 `optimal_labels_full`
- 跑 `VLT` 预测
- 跑 `TFT demand` 预测
- 生成正式对齐集

### 第四步：训练补货决策模型

如果第三步已经完成，并且你只是想直接训练决策模型，可以复用已有预测结果：

```bash
python run_full_data_prep.py --output_dir checkpoints/full_data_prep_full_pred --existing_labels_csv checkpoints/optimal_labels_full/optimal_replenishment_labels.csv --existing_labels_summary_csv checkpoints/optimal_labels_full/optimal_replenishment_summary.csv --keep_only_order_days --skip_vlt --skip_tft --train_decision
```

### 第五步：评估补货决策模型

```bash
python evaluate_replenishment_decision.py --aligned_csv checkpoints/full_data_prep_full_pred/decision_train_aligned.csv --split_csv checkpoints/full_data_prep_full_pred/decision_test_split.csv --labels_csv checkpoints/full_data_prep_full_pred/optimal_replenishment_labels.csv --model_path checkpoints/full_data_prep_full_pred/replenishment_mlp_aligned.pt --scaler_path checkpoints/full_data_prep_full_pred/decision_scaler_aligned.pt --output_dir checkpoints/full_data_prep_full_pred/decision_eval
```

### 第六步：分析高缺货成本组

```bash
python analyze_stockout_drivers.py --eval_dir checkpoints/full_data_prep_full_pred/decision_eval --split_csv checkpoints/full_data_prep_full_pred/decision_test_split.csv --output_dir checkpoints/full_data_prep_full_pred/decision_eval/stockout_analysis
```

---

## 7. 当前最值得关注的结果文件

如果你现在只想快速看项目核心结果，优先看这些：

### 7.1 标签

- [optimal_replenishment_labels.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/optimal_labels_full/optimal_replenishment_labels.csv)

### 7.2 正式训练集

- [decision_train_aligned.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/decision_train_aligned.csv)
- [decision_train_split.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/decision_train_split.csv)
- [decision_val_split.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/decision_val_split.csv)
- [decision_test_split.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/decision_test_split.csv)

### 7.3 预测结果

- [tft_demand_predictions.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/tft_demand_predictions.csv)
- [vlt_predictions.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/vlt_predictions.csv)

### 7.4 最终评估

- [decision_eval_metrics.json](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/decision_eval/decision_eval_metrics.json)
- [decision_group_costs.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/decision_eval/decision_group_costs.csv)

### 7.5 缺货驱动分析

- [group_diagnostics.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/decision_eval/stockout_analysis/group_diagnostics.csv)
- [topk_exclusion_comparison.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_full_pred/decision_eval/stockout_analysis/topk_exclusion_comparison.csv)

---

## 8. 现阶段的关键判断

当前这份代码在公开数据上已经完成了完整流程复现，但最终效果还不理想。

主要表现为：

- 决策模型训练流程已经跑通
- 评估脚本已经给出最终成本指标
- 与 `OPT` 相比，当前补货决策模型的总成本偏高
- 主要拖累来自 **缺货成本过高**

因此当前项目更准确的状态是：

**复现流程成功，模型效果一般，下一步更偏向模型改进而不是流程打通。**

---

## 9. 补充说明

- `optimal_label_generator.py` 主要吃 `CPU`
- `TFT / LSTM / MLP` 训练会受益于 `GPU`
- 如果上云，最推荐复用已有 `optimal_labels_full`，避免重复跑 `MILP`

---

## 10. 交接建议

如果你要在另一台电脑或 VS Code Codex 插件里继续做这个项目，建议先让它读：

- [README.md](D:/AA_postgraduate/thesis/code/my_project/README.md)
- [PROJECT_STATUS.md](D:/AA_postgraduate/thesis/code/my_project/PROJECT_STATUS.md)

其中：
- `README.md` 负责讲清项目结构和文件作用
- `PROJECT_STATUS.md` 负责讲清当前进度、结果和下一步建议
