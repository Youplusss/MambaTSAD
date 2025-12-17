# MambaTSAD: 基于 Mamba 的时间序列异常检测

MambaTSAD 是一个面向多变量时间序列异常检测（TSAD）的研究型实现，提供三种训练/推理分支：
- recon：重构式异常检测（多尺度重构误差）
- forecast：预测式异常检测（多步预测误差）
- hybrid：重构 + 预测混合模型（多任务联合训练与分数融合）
- hybrid_shared_adv：在 hybrid 基础上加入共享表征 + 对抗/鲁棒训练的变体（通过 `main_adv.py` 调用）

本仓库将数据处理、模型、训练与评估逻辑结构化在 `mambatsad/` 包中，并通过统一入口 `main.py`（标准分支）与 `main_adv.py`（shared_adv 分支）进行配置化训练与评估。


## 特性一览

- 支持多类典型时间序列异常检测数据集，已适配：
  - SMD（Server Machine Dataset，多机服务器日志）
  - MSL（NASA Mars Science Laboratory，多通道遥测）
  - SWaT（Secure Water Treatment，工业控制水处理系统）
  - WADI（Water Distribution，工业供水分配系统）
- 统一的滑动窗口切片与多实体（多机/多通道/多传感器）合并训练
- 重构分支：Bi-Mamba + 局部深度可分离卷积 + 多尺度金字塔重构
- 预测分支：倒置时间嵌入，在“变量维”使用 Mamba 建模变量间依赖，支持残差预测
- 混合模型：带不确定性参数的多任务加权（1/σ²），并在评估阶段融合两类分数
- shared_adv 变体：在混合模型基础上通过共享表征、对抗损失等机制进一步提升鲁棒性（见 `main_adv.py` 与 `hybrid_shared_adv`）
- 评估包含 point-adjust、AUC 及阈值搜索（F1 最大化）
- 训练过程记录到日志与 TensorBoard，并自动保存最优权重与可视化图


## 目录结构

```
MambaTSAD/
├── main.py                     # 统一训练/评估入口（标准 recon/forecast/hybrid）
├── main_adv.py                 # 对抗/鲁棒 shared_adv 入口
├── requirements.txt            # 依赖列表
├── data/                       # 原始或半处理数据（如 NASA MSL、SMD 文本等）
│   └── labeled_anomalies.csv   # NASA 风格标注文件示例
├── dataset/                    # 预处理后的统一格式数据根目录
│   ├── SMD/                    # SMD 预处理结果（train/test/test_label + 列表文件）
│   ├── MSL/                    # MSL 预处理结果
│   ├── SWaT/                   # SWaT ICS 数据（已按统一格式组织）
│   └── WADI/                   # WADI ICS 数据（已按统一格式组织）
├── mambatsad/
│   ├── data/                   # 数据集构建（SMD/MSL/SWaT/WADI after preprocess）
│   │   ├── __init__.py
│   │   ├── smd.py
│   │   ├── msl.py
│   │   └── ...                 # 若有 swat.py / wadi.py 等
│   ├── engine/
│   │   └── trainer.py          # 统一训练/评估循环（recon/forecast/hybrid/hybrid_shared_adv）
│   ├── models/                 # 模型定义（重构 / 预测 / 混合 / shared_adv）
│   │   ├── recon.py
│   │   ├── forecast.py
│   │   ├── hybrid.py
│   │   └── hybrid_shared_adv.py
│   └── utils/                  # 日志、指标、可视化、随机种子、对抗损失等
│       ├── logger.py
│       ├── metrics.py
│       ├── visualization.py
│       ├── adv_loss.py
│       └── seed.py
├── tools/                      # 原始数据 → 预处理格式 & 分析工具
│   ├── preprocess_smd.py
│   ├── preprocess_msl.py
│   ├── preprocess_smap.py
│   ├── preprocess_swat.py
│   ├── preprocess_wadi.py
│   ├── swat.py                 # SWaT 数据相关辅助脚本
│   ├── wadi.py                 # WADI 数据相关辅助脚本
│   └── separate_nasa_dataset.py
├── scripts/                    # Linux/macOS 下的示例脚本（.sh）
│   ├── run_smd_recon.sh
│   ├── run_smd_forecast.sh
│   ├── run_smd_hybrid.sh
│   ├── run_msl_recon.sh
│   ├── run_msl_forecast.sh
│   ├── run_msl_hybrid.sh
│   ├── run_smap_recon.sh
│   ├── run_smap_forecast.sh
│   ├── run_smap_hybrid.sh
│   ├── run_swat_recon.sh
│   ├── run_swat_forecast.sh
│   ├── run_swat_hybrid.sh
│   ├── run_wadi_recon.sh
│   ├── run_wadi_forecast.sh
│   ├── run_wadi_hybrid.sh
│   └── adv/                    # shared_adv 相关脚本
│       ├── run_smd_hybrid.sh   # 对应 main_adv + hybrid_shared_adv（SMD）
│       └── run_msl_hybrid.sh   # 对应 main_adv + hybrid_shared_adv（MSL）
└── logs/                       # 训练生成的日志、权重与图（运行后出现）
    ├── smd/
    ├── msl/
    ├── msl_adv_hybrid/
    ├── smd_all/
    ├── smd_best/
    ├── smd_forecast/
    ├── smd_hybrid/
    ├── smd_adv_hybrid/
    ├── smap_hybrid/
    ├── swat_hybrid/
    ├── wadi_hybrid/
    └── ...
```


## 环境与安装
- Python >= 3.8（建议 3.10/3.11）
- 推荐使用 GPU（支持 AMP 混合精度）

1) 创建虚拟环境（可选）
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安装依赖
- 直接安装（CPU/GPU 由 PyPI 上的 torch 构建决定；若需特定 CUDA 版本，请参考 PyTorch 官网安装说明）
```powershell
pip install -r requirements.txt
```
- 如需手动安装 PyTorch 指定 CUDA，请参考 https://pytorch.org/ 根据你的 CUDA 版本先安装 torch/torchvision/torchaudio，再执行：
```powershell
pip install -r requirements.txt --no-deps
```

依赖要点：
- mamba-ssm>=2.0.0（模型使用 Mamba 结构）
- torch>=2.1.0（建议与本机 CUDA 匹配的 wheel）


## 数据准备（预处理格式）

### 统一预处理格式约定

训练脚本期望已经预处理好的统一目录结构：
```text
processed_root/
  train/
    <entity_id>.npy       # 训练序列 (T_train, D)
  test/
    <entity_id>.npy       # 测试序列 (T_test, D)
  test_label/
    <entity_id>.npy       # 测试标签 (T_test,)
  entities.txt            # 实体列表（如机器/通道/工艺线 ID）
```
说明：
- `<entity_id>`：可代表服务器编号、传感器组合编号、过程段编号等。
- `entities.txt`：每行一个实体 ID，用于确定加载顺序与映射。

本仓库提供了将常见原始数据转换为上述格式的工具，并已在 `dataset/` 下为多类数据集组织了统一结构。

---

### SMD（Server Machine Dataset）

- 原始目录示例：`./data/SMD/train/*.txt`, `./data/SMD/test/*.txt`, `./data/SMD/test_label/*.txt`
- 预处理命令：
  ```powershell
  python tools/preprocess_smd.py --raw_root .\data\SMD --out_root .\dataset\SMD --use_global_scaler
  ```

执行后 `./dataset/SMD/` 结构类似：
```text
dataset/SMD/
  train/
    machine-1-1.npy
    ...
  test/
    machine-1-1.npy
    ...
  test_label/
    machine-1-1.npy
    ...
  entities.txt           # 机器列表
```

---

### MSL（NASA/telemanom 风格）

- 原始目录示例：
  ```text
  .\data\MSL\
    train\M-1.npy, M-2.npy, ...
    test\M-1.npy,  M-2.npy, ...
    labeled_anomalies.csv
  ```
- 预处理命令：
  ```powershell
  python tools/preprocess_msl.py --raw_root .\data\MSL --out_root .\dataset\MSL --use_global_scaler
  ```

- 可选：`tools/separate_nasa_dataset.py` 可辅助按 spacecraft（SMAP/MSL）拆分 telemanom 数据目录。

---

### SWaT / WADI 工业控制数据集说明

本仓库面向的 SWaT / WADI 数据集均来自工业控制（ICS）/SCADA 场景，是实际水处理与供水系统的多变量传感器时间序列：

- **SWaT（Secure Water Treatment）**
  - 来源：新加坡 SUTD 提供的实验性水处理过程系统数据集。
  - 场景：多阶段水处理工艺，包含水箱、水泵、阀门、流量、液位等多类传感器。
  - 常用于评估工业控制系统（ICS）异常检测模型。

- **WADI（Water Distribution）**
  - 来源：同一实验平台扩展的真实规模供水分配系统数据集。
  - 场景：城市供水分配网络，包含多个分配节点与管道流量/压力等传感器。
  - 常用于更大规模、长时序 ICS 异常检测评估。

> 版权与获取：
> 这两个数据集通常需要通过官方渠道（如论文作者或项目官网）申请和下载，本仓库不包含原始数据文件。请遵循原数据集的许可协议与使用条款。

---

### SWaT / WADI 在本仓库中的存放与结构

在 MambaTSAD 中，我们将 SWaT / WADI 转换并统一存放在 `dataset/SWaT` 与 `dataset/WADI` 目录下，格式与前述通用结构保持一致，便于与 SMD/MSL 复用同一训练代码。

典型目录树如下：

#### `dataset/SWaT` 结构示例

```text
dataset/SWaT/
  train/
    P1_train.npy
    P2_train.npy
    ...
  test/
    P1_test.npy
    P2_test.npy
    ...
  test_label/
    P1_label.npy
    P2_label.npy
    ...
  entities.txt            # 实体/过程段列表，例如 P1, P2, ...
```

#### `dataset/WADI` 结构示例

```text
dataset/WADI/
  train/
    S1_train.npy
    S2_train.npy
    ...
  test/
    S1_test.npy
    S2_test.npy
    ...
  test_label/
    S1_label.npy
    S2_label.npy
    ...
  entities.txt            # 实体/分区列表，例如 S1, S2, ...
```

说明：
- 具体文件名可能根据你的预处理脚本略有不同，但推荐保持「按实体/区域切分」的一致命名方式。
- `entities.txt` 决定训练和评估时各 `.npy` 文件的加载顺序。

---

### SWaT / WADI 的预处理与组织方式

根据你的数据来源不同，有两种常见方式将 SWaT/WADI 转换为统一格式：

1. **使用本仓库中的预处理脚本**

   - `tools/preprocess_swat.py`
   - `tools/preprocess_wadi.py`

   典型命令：

   ```powershell
   # SWaT
   python tools/preprocess_swat.py --raw_root .\data\SWaT --out_root .\dataset\SWaT --use_global_scaler

   # WADI
   python tools/preprocess_wadi.py --raw_root .\data\WADI --out_root .\dataset\WADI --use_global_scaler
   ```

   其中：
   - `--raw_root` 指向你解压后的原始数据目录（包含正常/攻击段 CSV 等）。
   - `--out_root` 生成的统一格式目录，即 `.\dataset\SWaT` / `.\dataset\WADI`。
   - `--use_global_scaler` 表示按全局统计量做标准化，便于多实体联合训练。

2. **手动组织到统一结构**

   若暂未使用上述脚本，你也可以自行将数据整理为前述统一结构：

   - 将每个“实体”（例如单条工艺线、一个水箱/管段、一组相关传感器）导出为一个 `.npy` 数组，形状为 `(T, D)`。
   - 将正常运行数据放入 `train/`，混合正常+攻击段数据放入 `test/`，对应标签（0/1）放入 `test_label/`。
   - 在 `entities.txt` 中逐行列出你的实体 ID，顺序需与 `train/`、`test/`、`test_label/` 中的 `.npy` 文件相对应。


## 快速开始（Windows PowerShell 示例）
以下示例假设你的预处理数据根目录为：
- SMD: `.\dataset\SMD`
- MSL: `.\dataset\MSL`
- SWaT: `.\dataset\SWaT`
- WADI: `.\dataset\WADI`

日志与模型将保存到 `--log_dir` 指定目录（若不存在会自动创建）。

### SMD 示例

- 重构分支（SMD）：
```powershell
python main.py --dataset smd --branch recon --processed_root .\dataset\SMD --log_dir .\logs\smd_recon --win_size 100 --batch_size 64 --epochs 50
```

- 预测分支（SMD）：
```powershell
python main.py --dataset smd --branch forecast --processed_root .\dataset\SMD --log_dir .\logs\smd_forecast --win_size 100 --pred_len 10 --batch_size 64 --epochs 50
```

- 混合分支（SMD）：
```powershell
python main.py --dataset smd --branch hybrid --processed_root .\dataset\SMD --log_dir .\logs\smd_hybrid --win_size 100 --pred_len 10 --batch_size 64 --epochs 50 --lr 1e-4 --weight_decay 5e-4 --lambda_recon 1.0 --lambda_forecast 1.0
```

### MSL 示例

- 重构分支（MSL）：
```powershell
python main.py --dataset msl --branch recon --processed_root .\dataset\MSL --log_dir .\logs\msl_recon --win_size 100 --batch_size 64 --epochs 50
```

- 预测分支（MSL）：
```powershell
python main.py --dataset msl --branch forecast --processed_root .\dataset\MSL --log_dir .\logs\msl_forecast --win_size 100 --pred_len 10 --batch_size 64 --epochs 50
```

- 混合分支（MSL）：
```powershell
python main.py --dataset msl --branch hybrid --processed_root .\dataset\MSL --log_dir .\logs\msl_hybrid --win_size 100 --pred_len 10 --batch_size 64 --epochs 50 --lr 1e-4 --weight_decay 5e-4 --lambda_recon 1.0 --lambda_forecast 1.0
```

### SWaT 示例（ICS 水处理场景）

假设你已经将 SWaT 数据集整理到 `.\dataset\SWaT`，结构符合前述约定。

- 重构分支（SWaT）：
```powershell
python main.py --dataset swat --branch recon --processed_root .\dataset\SWaT --log_dir .\logs\swat_recon --win_size 100 --batch_size 64 --epochs 50
```

- 预测分支（SWaT）：
```powershell
python main.py --dataset swat --branch forecast --processed_root .\dataset\SWaT --log_dir .\logs\swat_forecast --win_size 100 --pred_len 10 --batch_size 64 --epochs 50
```

- 混合分支（SWaT）：
```powershell
python main.py --dataset swat --branch hybrid --processed_root .\dataset\SWaT --log_dir .\logs\swat_hybrid --win_size 100 --pred_len 10 --batch_size 64 --epochs 50 --lr 1e-4 --weight_decay 5e-4 --lambda_recon 1.0 --lambda_forecast 1.0
```

### WADI 示例（ICS 供水分配场景）

假设你已经将 WADI 数据集整理到 `.\dataset\WADI`，结构符合前述约定。

- 重构分支（WADI）：
```powershell
python main.py --dataset wadi --branch recon --processed_root .\dataset\WADI --log_dir .\logs\wadi_recon --win_size 100 --batch_size 64 --epochs 50
```

- 预测分支（WADI）：
```powershell
python main.py --dataset wadi --branch forecast --processed_root .\dataset\WADI --log_dir .\logs\wadi_forecast --win_size 100 --pred_len 10 --batch_size 64 --epochs 50
```

- 混合分支（WADI）：
```powershell
python main.py --dataset wadi --branch hybrid --processed_root .\dataset\WADI --log_dir .\logs\wadi_hybrid --win_size 100 --pred_len 10 --batch_size 64 --epochs 50 --lr 1e-4 --weight_decay 5e-4 --lambda_recon 1.0 --lambda_forecast 1.0
```

> 提示：
> - 可以通过调整 `--win_size`、`--pred_len`、`--batch_size` 等超参数适配 SWaT/WADI 的时间粒度与序列长度。
> - 推荐 SWaT/WADI 采用与 SMD 相近的窗口和批大小作为起点，再根据 GPU 显存微调。

Linux/macOS 下可参考 `scripts/*.sh`。

---

### shared_adv 变体示例（使用 `main_adv.py`）

shared_adv 分支在标准 hybrid 模型的基础上，引入共享特征表示和对抗/鲁棒训练机制，入口为 `main_adv.py`，模型为 `hybrid_shared_adv`。

以下以 SMD、MSL 为例（若你在 SWaT/WADI 上也实现了 shared_adv，可类比使用相同参数）：

- SMD 上的 shared_adv 混合分支：
```powershell
python main_adv.py --dataset smd --branch hybrid_shared_adv --processed_root .\dataset\SMD --log_dir .\logs\smd_adv_hybrid --win_size 100 --pred_len 10 --batch_size 64 --epochs 50
```

- MSL 上的 shared_adv 混合分支：
```powershell
python main_adv.py --dataset msl --branch hybrid_shared_adv --processed_root .\dataset\MSL --log_dir .\logs\msl_adv_hybrid --win_size 100 --pred_len 10 --batch_size 64 --epochs 50
```

> 对应的 Linux/macOS 示例可参考：
> - `scripts/adv/run_smd_hybrid.sh`
> - `scripts/adv/run_msl_hybrid.sh`

## 命令行参数说明
来自 `main.py` / `main_adv.py`（仅列关键项）：
- 数据与日志
  - `--dataset {smd,msl,swat,wadi}`：数据集名称（需与对应数据加载器实现一致）
  - `--processed_root <PATH>`：预处理后的数据根目录（必填）
  - `--log_dir <PATH>`：日志与模型输出目录（默认 `./logs/exp`）
- 任务/窗口
  - `--branch {recon,forecast,hybrid,hybrid_shared_adv}`：训练分支（`hybrid_shared_adv` 需通过 `main_adv.py` 调用）
  - `--win_size`：窗口长度 L（默认 100）
  - `--pred_len`：预测步数 T_pred，仅在 forecast/hybrid/hybrid_shared_adv 下生效（默认 10；需满足 win_size > pred_len）
  - `--train_stride` / `--test_stride`：滑窗步长（默认 1）
- 训练超参
  - `--batch_size`（默认 64）、`--epochs`（默认 50）、`--lr`（默认 1e-3）、`--weight_decay`（默认 1e-4）
  - `--seed`（默认 42）、`--no_amp`（关闭 AMP，默认开启）、`--max_grad_norm`（梯度裁剪，默认 1.0）
  - `--patience`：F1 早停耐心轮数（默认 8）
- 混合/对抗分支系数
  - `--lambda_recon`（默认 1.0）、`--lambda_forecast`（默认 1.0）
  - 其余对抗/伪标签相关超参请参考 `main_adv.py -h`
- 评估
  - `--no_point_adjust`：关闭 point-adjust（默认开启）
  - `--num_workers`：DataLoader 后台线程数（默认 4）

查看完整帮助：
```powershell
python main.py -h
python main_adv.py -h
```


## 训练产物与日志
- 日志文件：`<log_dir>/MambaTSAD_YYYYmmdd_HHMMSS.log`
- TensorBoard：`<log_dir>/tb/`
- 最佳模型权重：`<log_dir>/best_model_<branch>.pt`
- 评估可视化图：`<log_dir>/scores_epoch{k}_<branch>.png`

F1 改善时自动保存最优权重；若连续 `--patience` 个 epoch 无提升则早停。


## 评估与分数
- 重构分支：窗口内重构误差（MSE）在时间维聚合，并在原始序列上做覆盖平均
- 预测分支：仅对窗口的“预测片段（context_len 之后的 pred_len 部分）”计算预测 MSE
- 混合分支：对重构/预测分数进行 z-score 归一化后，以不确定性/λ 形成权重并融合
- 自动在一组候选阈值上搜索 F1 最优阈值；可选启用 point-adjust


## 常见问题（FAQ）
- mamba-ssm 安装失败或太慢？
  - 先确认 Python 与编译环境；或使用预编译 wheel；必要时在 Linux 上安装更顺畅。
- CUDA/显卡不可用？
  - 代码可在 CPU 上运行但训练较慢；建议按 PyTorch 官网指引安装与你 CUDA 版本匹配的 torch。
- 预测分支报 “win_size 必须大于 pred_len”？
  - 确保设置 `--win_size > --pred_len`，例如 `--win_size 100 --pred_len 10`。
- 数据预处理后仍有 NaN/Inf？
  - 训练/测试前均有防守式清洗（nan_to_num + clamp），并在全局 z-score 上统一归一化；若仍异常，请检查原始数据质量。
- SWaT/WADI 训练效果不稳定？
  - 这类 ICS 数据通常存在长时间段的稳定运行与少量攻击段，建议适当调小 `--train_stride` / `--test_stride`、增大 `--epochs`，并观察日志中的 F1、AUC 曲线。


## 参考/致谢
- Mamba: Structured State Space for Sequence Modeling（mamba-ssm）
- MambaAD / LSS 思想与时间序列改造
- 多任务不确定性加权（Kendall et al., 2018）
- SWaT / WADI 工业控制数据集及相关论文/项目


## 许可证
本项目仅供研究与学习使用，具体以仓库 LICENSE 为准（如未附带，则默认保留所有权利）。
