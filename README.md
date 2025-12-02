MambaTSAD/
├── README.md
├── requirements.txt
├── main_smd.py              # 训练 + 测试 SMD 的入口
├── mambatsad/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── mambatsad_ts.py  # 模型主体：Bi-Mamba + LSS + 金字塔重构
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── smd.py           # SMD 数据集处理（重点完整给出）
│   │   ├── msl.py           # MSL 数据集处理（基于 *.npy 格式）
│   │   ├── swat.py          # SWaT 数据集处理（CSV + label 列）
│   │   └── wadi.py          # WADI 数据集处理（CSV + label 列）
│   └── utils/
│       ├── logger.py        # 日志
│       ├── metrics.py       # F1 / Precision / Recall 等
│       └── visualization.py # 结果可视化
└── scripts/
    ├── run_smd.sh           # 一键启动 SMD 训练/测试
    └── run_smd_debug.sh     # 快速 debug 配置
