# MambaTSAD: 基于 Mamba 的时间序列异常检测

MambaTSAD 是一个面向多变量时间序列异常检测（TSAD）的研究型实现，提供三种训练/推理分支：
- recon：重构式异常检测（多尺度重构误差）
- forecast：预测式异常检测（多步预测误差）
- hybrid：重构 + 预测混合模型（多任务联合训练与分数融合）

本仓库将数据处理、模型、训练与评估逻辑结构化在 `mambatsad/` 包中，并通过统一入口 `main.py` 进行配置化训练与评估。


## 特性一览
- 支持 SMD（Server Machine Dataset，多机）与 MSL（NASA Multi-channel）两类经典数据集（经预处理为一致格式）
- 统一的滑动窗口切片与多实体（多机/多通道）合并训练
- 重构分支：Bi-Mamba + 局部深度可分离卷积 + 多尺度金字塔重构
- 预测分支：倒置时间嵌入，在“变量维”使用 Mamba 建模变量间依赖，支持残差预测
- 混合模型：带不确定性参数的多任务加权（1/σ²），并在评估阶段融合两类分数
- 评估包含 point-adjust、AUC 及阈值搜索（F1 最大化）
- 训练过程记录到日志与 TensorBoard，并自动保存最佳权重与可视化图


## 目录结构
```
MambaTSAD/
├── main.py                     # 统一训练/评估入口
├── requirements.txt            # 依赖列表
├── mambatsad/
│   ├── data/                   # 数据集构建（SMD/MSL after preprocess）
│   │   ├── __init__.py
│   │   ├── smd.py
│   │   └── msl.py
│   ├── engine/
│   │   └── trainer.py          # 统一训练/评估循环（recon/forecast/hybrid）
│   ├── models/                 # 模型定义（重构 / 预测 / 混合）
│   │   ├── recon.py
│   │   ├── forecast.py
│   │   └── hybrid.py
│   └── utils/                  # 日志、指标、可视化、随机种子等
│       ├── logger.py
│       ├── metrics.py
│       ├── visualization.py
│       └── seed.py
├── tools/                      # 原始数据 → 预处理格式
│   ├── preprocess_smd.py
│   ├── preprocess_msl.py
│   └── separate_nasa_dataset.py
├── scripts/                    # Linux/macOS 下的示例脚本（.sh）
│   ├── run_smd_recon.sh
│   ├── run_smd_forecast.sh
│   ├── run_smd_hybrid.sh
│   ├── run_msl_recon.sh
│   ├── run_msl_forecast.sh
│   └── run_msl_hybrid.sh
└── logs/                       # 训练生成的日志、权重与图（运行后出现）
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
训练脚本期望已经预处理好的统一目录结构：
```
processed_root/
  train/
    <entity_id>.npy     # 训练序列 (T_train, D)
  test/
    <entity_id>.npy     # 测试序列 (T_test, D)
  test_label/
    <entity_id>.npy     # 测试标签 (T_test,)
  machines.txt | channels.txt   # 实体列表（SMD: machines.txt；MSL: channels.txt）
```

本仓库提供了将常见原始数据转换为上述格式的工具：

- SMD（ServerMachineDataset）
  - 原始目录示例：`./data/SMD/train/*.txt`, `./data/SMD/test/*.txt`, `./data/SMD/test_label/*.txt`
  - 预处理命令：
    ```powershell
    python tools/preprocess_smd.py --raw_root .\data\SMD --out_root .\dataset\SMD --use_global_scaler
    ```

- MSL（NASA/telemanom 风格）
  - 原始目录示例：
    ```
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


## 快速开始（Windows PowerShell 示例）
以下示例假设你的预处理数据根目录为：
- SMD: `.\u200bdataset\SMD`
- MSL: `.\u200bdataset\MSL`

日志与模型将保存到 `--log_dir` 指定目录（若不存在会自动创建）。

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

- 将 `--dataset msl` 与 `--processed_root .\dataset\MSL` 替换即可在 MSL 数据上运行。Linux/macOS 下可参考 `scripts/*.sh`。


## 命令行参数说明
来自 `main.py`（仅列关键项）：
- 数据与日志
  - `--dataset {smd,msl}`：数据集名称（必选其一）
  - `--processed_root <PATH>`：预处理后的数据根目录（必填）
  - `--log_dir <PATH>`：日志与模型输出目录（默认 `./logs/exp`）
- 任务/窗口
  - `--branch {recon,forecast,hybrid}`：训练分支
  - `--win_size`：窗口长度 L（默认 100）
  - `--pred_len`：预测步数 T_pred，仅在 forecast/hybrid 下生效（默认 10；需满足 win_size > pred_len）
  - `--train_stride` / `--test_stride`：滑窗步长（默认 1）
- 训练超参
  - `--batch_size`（默认 64）、`--epochs`（默认 50）、`--lr`（默认 1e-3）、`--weight_decay`（默认 1e-4）
  - `--seed`（默认 42）、`--no_amp`（关闭 AMP，默认开启）、`--max_grad_norm`（梯度裁剪，默认 1.0）
  - `--patience`：F1 早停耐心轮数（默认 8）
- 混合分支系数
  - `--lambda_recon`（默认 1.0）、`--lambda_forecast`（默认 1.0）
- 评估
  - `--no_point_adjust`：关闭 point-adjust（默认开启）
  - `--num_workers`：DataLoader 后台线程数（默认 4）

查看完整帮助：
```powershell
python main.py -h
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


## 参考/致谢
- Mamba: Structured State Space for Sequence Modeling（mamba-ssm）
- MambaAD / LSS 思想与时间序列改造
- 多任务不确定性加权（Kendall et al., 2018）


## 许可证
本项目仅供研究与学习使用，具体以仓库 LICENSE 为准（如未附带，则默认保留所有权利）。
