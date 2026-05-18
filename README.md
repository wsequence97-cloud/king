# TFT-MPIR Reproduction Project

本项目用于复现论文 **TFT-MPIR: An end-to-end multi-period inventory replenishment strategy based on temporal fusion transformer**，并基于公开数据集逐步补齐完整流程。

当前项目已经覆盖：
- `Demand Forecast`：基于 `TFT` 的需求预测
- `VLT Forecast`：基于 `LSTM` 的提前期预测
- `Post-hoc Optimal Labeling`：基于 `MILP` 的最优补货标签生成
- `Replenishment Decision`：基于 `MLP` 的补货决策训练
- `Full Data Prep`：一键完成标签生成、特征构造、训练集对齐与切分

---

## 1. 项目结构

### 核心代码

- [train.py](D:/AA_postgraduate/thesis/code/my_project/train.py)
  现有的 `TFT` 需求预测训练入口，使用 `sales_data.csv` 训练 demand forecast 模型。

- [tft_model.py](D:/AA_postgraduate/thesis/code/my_project/tft_model.py)
  `TFT` 模型定义文件，供 `train.py` 和后续 `full_pipeline.py` 调用。

- [data_preprocessing.py](D:/AA_postgraduate/thesis/code/my_project/data_preprocessing.py)
  需求预测模块的数据预处理逻辑，负责构建 `DC-SKU` 的时序窗口样本。

- [dataset.py](D:/AA_postgraduate/thesis/code/my_project/dataset.py)
  将预处理后的样本包装为 PyTorch `Dataset / DataLoader`。

- [visualize.py](D:/AA_postgraduate/thesis/code/my_project/visualize.py)
  训练过程可视化与训练曲线保存。

- [full_pipeline.py](D:/AA_postgraduate/thesis/code/my_project/full_pipeline.py)
  当前最重要的主流程文件，整合了：
  - `VLT` 预测模块
  - 接入现有 `TFT checkpoint` 的 demand forecast 推理
  - 补货决策特征构造
  - 补货 `MLP` 训练
  - 对齐最优标签后的补货训练入口

- [optimal_label_generator.py](D:/AA_postgraduate/thesis/code/my_project/optimal_label_generator.py)
  基于论文 `post-hoc optimal solution` 思路构建的最优补货标签生成器。
  使用真实历史 `sales + leadtime + inventory + tariff + unit_rate`，通过 `MILP` 求解最优补货量 `q_t*`。

- [run_full_data_prep.py](D:/AA_postgraduate/thesis/code/my_project/run_full_data_prep.py)
  一键式数据准备脚本，适合上云运行。可完成：
  - 全量最优标签生成
  - `VLT` 预测
  - `TFT demand` 预测
  - 补货训练特征构造
  - 与最优标签对齐
  - `train / val / test` 切分
  - 可选直接训练补货决策模型

---

## 2. 数据文件

数据位于 [data](D:/AA_postgraduate/thesis/code/my_project/data)：

- `sales_data.csv`
  历史销量数据，需求预测模块的核心输入。

- `leadtime_data.csv`
  历史 `VLT` 数据，提前期预测模块的核心输入。

- `dc_inventory.csv`
  各 `DC-SKU` 初始库存。

- `factory_inventory.csv`
  各工厂 `SKU` 初始库存。

- `dc_capacity.csv`
  各 `DC` 库容。

- `push_limit.csv`
  工厂发货上下限。

- `transport_tariff.csv`
  工厂到 `DC` 的单位运输成本。

- `unit_rate.csv`
  `SKU` 的托盘-箱数转换关系及体积信息。

---

## 3. 论文与辅助材料

- [Guo 等 - 2025 - TFT-MPIR An end-to-end multi-period inventory replenishment strategy based on temporal fusion trans.pdf](D:/AA_postgraduate/thesis/code/my_project/Guo%20等%20-%202025%20-%20TFT-MPIR%20An%20end-to-end%20multi-period%20inventory%20replenishment%20strategy%20based%20on%20temporal%20fusion%20trans.pdf)
  复现目标论文。

- [data_exp.md](D:/AA_postgraduate/thesis/code/my_project/data_exp.md)
  数据说明整理。

- `image.png` 到 `image-7.png`
  数据字段说明截图，来自数据说明文档。

- [paper_extract_p3_p5.txt](D:/AA_postgraduate/thesis/code/my_project/paper_extract_p3_p5.txt)
- [paper_extract_p7_p11.txt](D:/AA_postgraduate/thesis/code/my_project/paper_extract_p7_p11.txt)
  从论文 PDF 中抽取的文本片段，用于本地分析目标函数、约束和实验设定。

---

## 4. checkpoints 目录

[checkpoints](D:/AA_postgraduate/thesis/code/my_project/checkpoints) 目录保存训练权重、推理结果和烟测产物。

### 现有文件

- `best_model.pt`
  当前 demand forecast 最优权重。

- `last_checkpoint.pt`
  当前 demand forecast 完整 checkpoint，包含 `model_cfg`，可用于恢复 TFT 结构和参数。

- `training_curve.png`
  需求预测训练曲线。

### 烟测目录

- `full_pipeline_smoke/`
  `VLT + 决策模块` 的初步烟测结果。

- `full_pipeline_tft_smoke/`
  接入真实 `TFT` 预测后的烟测结果。

- `optimal_labels_smoke/`
- `optimal_labels_smoke20/`
  最优标签生成器的小批量 / 中批量验证结果。

- `full_data_prep_smoke/`
  一键式数据准备脚本的烟测结果，里面已经有：
  - `optimal_replenishment_labels.csv`
  - `decision_features_proxy.csv`
  - `decision_train_aligned.csv`
  - `decision_train_split.csv`
  - `decision_val_split.csv`
  - `decision_test_split.csv`
  - `run_metadata.json`

---

## 5. 当前推荐运行顺序

### 第一步：训练或确认需求预测模型

如果 `checkpoints/best_model.pt` 和 `checkpoints/last_checkpoint.pt` 已可用，可以直接跳过。

否则运行：

```bash
python train.py --data data/sales_data.csv
```

### 第二步：生成最优补货标签

```bash
python optimal_label_generator.py --output_dir checkpoints/optimal_labels_full
```

### 第三步：一键生成训练数据并对齐

```bash
python run_full_data_prep.py --output_dir checkpoints/full_data_prep_full --keep_only_order_days
```

### 第四步：一键完成对齐并训练补货模型

```bash
python run_full_data_prep.py --output_dir checkpoints/full_data_prep_full --keep_only_order_days --train_decision
```

---

## 6. 适合上云的命令

### 只做全量标签生成和训练集对齐

```bash
python run_full_data_prep.py --output_dir checkpoints/full_data_prep_full --keep_only_order_days
```

### 跑完整链路

```bash
python run_full_data_prep.py --output_dir checkpoints/full_data_prep_full --keep_only_order_days --train_decision
```

### 调试时先缩小规模

```bash
python run_full_data_prep.py --limit_groups 10 --skip_vlt --skip_tft --output_dir checkpoints/debug_prep --keep_only_order_days
```

---

## 7. 当前实现边界

当前项目已经可以跑通完整实验链路，但有两点需要注意：

- `optimal_label_generator.py` 生成的是基于论文成本结构和公开数据的 `post-hoc optimal labels`，这是训练补货决策模块所需的监督标签。
- `MILP` 标签生成主要吃 `CPU`，不是 `GPU`；而 `TFT / LSTM / MLP` 训练会从 `GPU` 中明显受益。

---

## 8. 你现在可以把它理解成什么状态

这个仓库原本主要只有 `TFT demand forecast`。

现在已经扩展成一个完整的复现工程，能够覆盖：
- 论文数据读取
- `Demand Forecast`
- `VLT Forecast`
- 最优补货标签求解
- 补货训练集构造与对齐
- 补货决策训练

如果后面继续推进，最自然的下一步是：
- 在云服务器上跑全量标签生成
- 跑完整 `VLT + TFT + aligned decision` 训练
- 再根据论文指标做系统评估
