# SAM 两两比较器

对咔唑/膦酸类 SAM 分子做相对排序：输入两个 SMILES，输出
`P(A 优于 B)`。模型不预测绝对 PCE。

## 模型

- 整分子 D-MPNN：Chemprop 2 消息传递 + 平均池化，3 个随机种子。
- Graphormer 2D：元素/度数/氢数/芳香性/电荷节点特征，最短路径距离
  偏置与路径键类型偏置，3 个随机种子。
- 比较头为无偏置的 `sigmoid(head(E_B - E_A))`，因此
  `P(A>B) + P(B>A) = 1`，同一分子自比为 0.5。
- 最终概率先对每种架构的 3 个种子平均，再对两种架构平均。

仓库中的 `checkpoints/` 是通过分子隔离 5 折交叉验证与 Optuna TPE +
HyperbandPruner 调参后，使用全部 292 个分子重新训练的 6 个最终模型。

## 快速开始

已验证环境：

```text
Python 3.13.16
chemprop 2.2.2
torch 2.14.1
rdkit 2026.9.1
numpy 2.5.4
pandas 3.0.6
openpyxl 3.1.5
scikit-learn 1.9.1
optuna 5.0.0
```

安装：

```powershell
python -m pip install -r requirements.txt
```

比较两个分子：

```powershell
python src\predict.py "SMILES_A" "SMILES_B"
```

机器可读 JSON：

```powershell
python src\predict.py "SMILES_A" "SMILES_B" --json
```

基准排序与候选初筛：

```powershell
python src\evaluate.py
python src\evaluate.py --screen data\候选库.csv
```

`run_eval.bat` 会依次查找 `%SAM_PYTHON%`、项目内 `.venv`、当前用户的
`chemprop2` Conda 环境和系统 `python`。

## 目录

```text
checkpoints/          最终 6 模型与 ensemble_config.json
data/                 292 个分子清单、17,184 个无序伪配对
cv/                   5 折划分、OOF 预测、调参记录与汇总指标
src/                  D-MPNN/Graphormer 模型、推理、评估、训练
results/              新模型的基准矩阵与指标
requirements.txt      已验证依赖
run_eval.bat          一键基准评估
run_cv_train.bat      GPU 交叉验证训练入口
```

## 数据

- `data/合并分子清单_292.xlsx`：292 个分子的 SMILES 与 `PCE_max`。
- `data/pseudo_pairs.csv`：17,184 个无序配对的正反双向共 34,368 行。
  `y=1` 表示 `smiles_A` 的 `PCE_max` 更高。
- 训练标签逐条与 `PCE_max` 一致，PCE 完全相同的组合不参与训练。

## 五折交叉验证

验证采用冷启动目标分子策略：每个无序配对只评估一次，被查询的目标
分子属于当前留出折，绝不参与该折训练；配对中的参照分子可以是训练
分子。脚本同时报告两端分子都未出现的严格子集指标。

完整 17,184 个配对的 OOF 结果：

| 指标 | 冷启动目标分子 | 严格双端未见 |
|---|---:|---:|
| 配对数量 | 17,184 | 3,424 |
| AUC | 0.9601 | 0.9162 |
| 0.5 阈值准确率 | 0.8994 | 0.8420 |
| Brier | 0.0764 | 0.1184 |

逐折冷启动 AUC 为 `0.9643 / 0.9924 / 0.9670 / 0.8700 / 0.9799`，
均值 `0.9547 ± 0.0435`。折 3 的严格未见 AUC 为 `0.7817`，说明
模型对部分分子分组的泛化仍有限，不应宣称对新骨架普遍接近 0.96。

详细记录见 [cv/README.md](cv/README.md) 和
[cv/summary.json](cv/summary.json)。

## 重新训练

### 与原配置一致的完整重训

```powershell
python src\train.py --model both --epochs 600 --seeds 0 1 2 --outdir checkpoints_retrained
```

### GPU 交叉验证训练

GPU 环境：

```powershell
conda create -n chemprop2-gpu --clone chemprop2 -y
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install optuna==5.0.0
```

完整训练：

```powershell
run_cv_train.bat --epochs 600 --folds 5 --seeds 0 1 2 `
  --models wo gt --trials-per-fold 8 --eval-interval 20 `
  --device cuda --outdir runs\sam_cv_5fold
```

每个外层折只使用其余训练分子做内层分子隔离调参。Optuna 使用 TPE
采样器和 HyperbandPruner，最多 8 个试验，最终配置训练 600 轮。
分折、调参、检查点、OOF 预测与指标全部写入独立运行目录。中断后可
用 `--resume` 和相同 `--outdir` 继续。

## 输入约束与适用边界

- SMILES 必须可被 RDKit 解析。
- 仅支持训练覆盖的元素集合。
- Graphormer 最多支持 112 个重原子，超限会直接报错。
- 适用域为与训练数据相近的咔唑/膦酸类 SAM。
- 输出概率用于排序，不应解释为绝对 PCE 或校准后的实验置信度。
- “新分子 vs 已知分子”的排序通常比两个全新骨架互比更可靠。

