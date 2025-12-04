import os
import pandas as pd
import shutil
import glob


def separate_nasa_dataset():
    """精确分离NASA MSL/SMAP数据集"""

    data_dir = r"./data/MSL/"
    output_dir = r"./dataset/"

    # 打印关键路径
    print("=" * 60)
    print(f"原始数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 标签文件路径
    labels_path = os.path.join(data_dir, "labeled_anomalies.csv")
    print(f"标签文件路径: {labels_path}")

    # 确保标签文件存在
    if not os.path.exists(labels_path):
        print(f"错误: 未找到标签文件 {labels_path}")
        return

    # 加载标签文件
    try:
        labels_df = pd.read_csv(labels_path, dtype=str)
        print(f"成功加载 {len(labels_df)} 条标签记录")
        print(f"列名: {labels_df.columns.tolist()}")
    except Exception as e:
        print(f"加载标签文件失败: {str(e)}")
        return

    # 创建航天器目录结构
    for spacecraft in ['SMAP', 'MSL']:
        for split in ['train', 'test']:
            os.makedirs(os.path.join(output_dir, spacecraft, split), exist_ok=True)

    # 统计信息
    counters = {
        'SMAP_train': 0,
        'SMAP_test': 0,
        'MSL_train': 0,
        'MSL_test': 0
    }

    # 处理每个通道
    print("\n开始处理通道数据...")
    for idx, row in labels_df.iterrows():
        channel_id = row['chan_id']
        spacecraft = row['spacecraft']

        # 确保航天器类型有效
        if spacecraft not in ['SMAP', 'MSL']:
            print(f"警告: 通道 {channel_id} 有无效的航天器类型: {spacecraft}")
            continue

        # 分别在测试集和训练集中查找
        found_in_test = False
        found_in_train = False

        # 在测试集中查找
        test_path = os.path.join(data_dir, "test", f"{channel_id}.npy")
        if os.path.exists(test_path):
            dest_dir = os.path.join(output_dir, spacecraft, "test")
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, f"{channel_id}.npy")
            shutil.copy2(test_path, dest_path)
            counters[f"{spacecraft}_test"] += 1
            print(f"复制测试集文件 {channel_id} → {spacecraft}/test")
            found_in_test = True

        # 在训练集中查找
        train_path = os.path.join(data_dir, "train", f"{channel_id}.npy")
        if os.path.exists(train_path):
            dest_dir = os.path.join(output_dir, spacecraft, "train")
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, f"{channel_id}.npy")
            shutil.copy2(train_path, dest_path)
            counters[f"{spacecraft}_train"] += 1
            print(f"复制训练集文件 {channel_id} → {spacecraft}/train")
            found_in_train = True

        if not found_in_test and not found_in_train:
            print(f"警告: 未找到通道 {channel_id} 的数据文件")

    # 打印结果
    print("\n" + "=" * 60)
    print("NASA数据集分离完成 - 详细结果")
    print("=" * 60)
    print(f"总通道数: {len(labels_df)}")
    print(f"SMAP数据集:")
    print(f"  训练集通道: {counters['SMAP_train']}")
    print(f"  测试集通道: {counters['SMAP_test']}")
    print(f"  总计通道: {counters['SMAP_train'] + counters['SMAP_test']}")
    print(f"\nMSL数据集:")
    print(f"  训练集通道: {counters['MSL_train']}")
    print(f"  测试集通道: {counters['MSL_test']}")
    print(f"  总计通道: {counters['MSL_train'] + counters['MSL_test']}")
    print("=" * 60)

    # 保存处理日志
    log_path = os.path.join(output_dir, "processing_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"分离时间: {pd.Timestamp.now()}\n")
        log.write(f"原始数据目录: {data_dir}\n")
        log.write(f"输出目录: {output_dir}\n\n")
        log.write(f"总通道数: {len(labels_df)}\n")
        log.write(f"SMAP训练集通道: {counters['SMAP_train']}\n")
        log.write(f"SMAP测试集通道: {counters['SMAP_test']}\n")
        log.write(f"MSL训练集通道: {counters['MSL_train']}\n")
        log.write(f"MSL测试集通道: {counters['MSL_test']}\n")

    print(f"\n处理日志已保存至: {log_path}")

    # 打印最终目录结构
    print("\n输出目录结构:")
    print(f"{output_dir}/")
    print(f"├── SMAP/")
    print(f"│   ├── train/   # {counters['SMAP_train']}个npy文件")
    print(f"│   └── test/    # {counters['SMAP_test']}个npy文件")
    print(f"└── MSL/")
    print(f"    ├── train/   # {counters['MSL_train']}个npy文件")
    print(f"    └── test/    # {counters['MSL_test']}个npy文件")

    print("\n处理完成!")


if __name__ == "__main__":
    import pandas as pd

    separate_nasa_dataset()