# Project Status

## 项目目标

本项目用于复现论文 **TFT-MPIR: An end-to-end multi-period inventory replenishment strategy based on temporal fusion transformer**，目标是补齐完整三大模块：

- `Demand Forecast`：预测未来销量
- `VLT Forecast`：预测补货提前期
- `Replenishment Decision`：结合前两者输出补货量

---

## 当前完成情况

### 1. Demand Forecast

已完成基于 `TFT` 的需求预测模块。

相关文件：
- [train.py](D:/AA_postgraduate/thesis/code/my_project/train.py)
- [tft_model.py](D:/AA_postgraduate/thesis/code/my_project/tft_model.py)
- [data_preprocessing.py](D:/AA_postgraduate/thesis/code/my_project/data_preprocessing.py)
- [dataset.py](D:/AA_postgraduate/thesis/code/my_project/dataset.py)

现有权重：
- [best_model.pt](D:/AA_postgraduate/thesis/code/my_project/checkpoints/best_model.pt)
- [last_checkpoint.pt](D:/AA_postgraduate/thesis/code/my_project/checkpoints/last_checkpoint.pt)

说明：
- `last_checkpoint.pt` 里包含 `model_cfg`
- `full_pipeline.py` 已经可以直接读取该 checkpoint 做真实 TFT 推理

### 2. VLT Forecast

已完成基于 `LSTM` 的 `VLT` 预测模块，并整合在：
- [full_pipeline.py](D:/AA_postgraduate/thesis/code/my_project/full_pipeline.py)

功能包括：
- 构造 `leadtime_data.csv` 的时序样本
- 训练 `LSTM`
- 保存 `vlt_predictions.csv`

### 3. 补货决策模块

已完成基于 `MLP` 的补货决策模块，并整合在：
- [full_pipeline.py](D:/AA_postgraduate/thesis/code/my_project/full_pipeline.py)

功能包括：
- 接入真实 `TFT demand forecast`
- 接入 `VLT forecast`
- 构造补货特征
- 训练补货决策模型

### 4. 最优标签生成

已完成基于论文 `post-hoc optimal solution` 思路的最优标签生成器：
- [optimal_label_generator.py](D:/AA_postgraduate/thesis/code/my_project/optimal_label_generator.py)

功能包括：
- 使用真实历史 `sales + leadtime + inventory + tariff + unit_rate`
- 建立 `MILP`
- 为每个 `DC-SKU-date` 生成最优补货标签 `q_t*`

### 5. 一键式数据准备

已完成一键式数据准备脚本：
- [run_full_data_prep.py](D:/AA_postgraduate/thesis/code/my_project/run_full_data_prep.py)

功能包括：
- 生成最优标签
- 可选生成 `TFT demand` 预测
- 可选生成 `VLT` 预测
- 构造补货特征表
- 与最优标签对齐
- 自动切分 `train / val / test`
- 可选直接训练补货决策模型

---

## 当前关键结果

### 全量最优标签生成结果

已经成功运行：

```bash
python optimal_label_generator.py --output_dir checkpoints/optimal_labels_full --resume --time_limit 60
```

当前结果如下：

- `holding_cost_ratio = 0.01`
- `stockout_to_transport_ratio = 1.5`
- `order_weekdays = [0, 4]`
- `time_limit = 60`
- `n_rows = 73959`
- `n_groups = 831`
- `n_failed_groups = 1`
- `total_cost_sum = 4929958066.090885`

结果目录：
- [optimal_labels_full](D:/AA_postgraduate/thesis/code/my_project/checkpoints/optimal_labels_full)

关键文件：
- `optimal_replenishment_labels.csv`
- `optimal_replenishment_summary.csv`
- `failed_groups.csv`
- `metadata.json`

说明：
- 全量标签已经基本可用
- 只有 `1` 个失败组
- 不需要因为这 `1` 个失败组暂停整体实验

### 已有烟测目录

- [full_pipeline_smoke](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_pipeline_smoke)
- [full_pipeline_tft_smoke](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_pipeline_tft_smoke)
- [optimal_labels_smoke](D:/AA_postgraduate/thesis/code/my_project/checkpoints/optimal_labels_smoke)
- [optimal_labels_smoke20](D:/AA_postgraduate/thesis/code/my_project/checkpoints/optimal_labels_smoke20)
- [full_data_prep_smoke](D:/AA_postgraduate/thesis/code/my_project/checkpoints/full_data_prep_smoke)

这些目录说明：
- `VLT` 模块已经能训练
- `TFT` 预测已经能接入决策特征
- 标签生成器已验证
- 特征表和标签已能成功对齐

---

## 当前推荐下一步

当前最自然的后续步骤是：

### 1. 先做全量训练集对齐

推荐命令：

```bash
python run_full_data_prep.py --output_dir checkpoints/full_data_prep_full --keep_only_order_days
```

作用：
- 使用 `optimal_labels_full`
- 构造补货决策训练特征
- 对齐最优标签
- 生成训练/验证/测试切分

### 2. 再训练补货决策模块

推荐命令：

```bash
python run_full_data_prep.py --output_dir checkpoints/full_data_prep_full --keep_only_order_days --train_decision
```

作用：
- 在完成对齐后直接训练补货 `MLP`

---

## 关键脚本说明

### [train.py](D:/AA_postgraduate/thesis/code/my_project/train.py)

作用：
- 单独训练 demand forecast 的 `TFT`

### [full_pipeline.py](D:/AA_postgraduate/thesis/code/my_project/full_pipeline.py)

作用：
- 训练 `VLT`
- 读取 `TFT checkpoint`
- 生成 demand forecast
- 构造补货特征
- 训练补货 `MLP`

### [optimal_label_generator.py](D:/AA_postgraduate/thesis/code/my_project/optimal_label_generator.py)

作用：
- 用 `MILP` 生成最优补货标签

注意：
- 该脚本主要吃 `CPU`
- 不是 `GPU` 重点收益模块

### [run_full_data_prep.py](D:/AA_postgraduate/thesis/code/my_project/run_full_data_prep.py)

作用：
- 一键打通“标签生成 + 预测 + 对齐 + 切分 + 可选训练”

---

## 已知问题与注意事项

### 1. Python 环境兼容性

当前环境偏旧，之前已经针对部分文件做过类型注解兼容处理，尤其是：
- [tft_model.py](D:/AA_postgraduate/thesis/code/my_project/tft_model.py)

如果换机器后 Python 版本不同，需要优先确认导入是否正常。

### 2. 标签生成吃 CPU

`optimal_label_generator.py` 是 `MILP` 求解，不依赖 GPU。

所以：
- 上云跑标签生成时主要看 CPU 性能
- 上云跑 `TFT/LSTM/MLP` 训练时 GPU 更重要

### 3. 失败组处理

全量标签目前仅失败 `1` 个组，优先级不高。

如果后续需要补跑，可以读取：
- [failed_groups.csv](D:/AA_postgraduate/thesis/code/my_project/checkpoints/optimal_labels_full/failed_groups.csv)

---

## 给下一位 Codex / 下一台电脑的简短交接说明

可以直接复制下面这段作为开场上下文：

```text
这个项目是复现 TFT-MPIR 论文的。
当前进度：
1. TFT demand forecast 已完成，checkpoint 在 checkpoints/best_model.pt 和 checkpoints/last_checkpoint.pt。
2. VLT 预测模块、补货决策模块、full pipeline 都已经实现。
3. optimal_label_generator.py 已经跑过全量标签，输出在 checkpoints/optimal_labels_full。
4. 当前全量标签结果：n_rows=73959, n_groups=831, n_failed_groups=1。
5. 下一步希望继续做训练集对齐和补货决策训练。
请先阅读 README.md 和 PROJECT_STATUS.md，再继续工作。
```

---

## 当前状态一句话总结

项目已经从“只完成 demand forecast”推进到“已具备完整复现链路”，并且全量最优补货标签已经基本生成完成，当前可以直接进入训练集对齐和补货决策训练阶段。
