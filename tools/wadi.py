import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# max min(0-1)
def norm(train, test):
    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train)  # scale training data to [0,1] range
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)

    return train_ret, test_ret


# downsample by 10
def downsample(data, labels, down_len):
    np_data = np.array(data)
    np_labels = np.array(labels)

    orig_len, col_num = np_data.shape

    down_time_len = orig_len // down_len

    np_data = np_data.transpose()
    # print('before downsample', np_data.shape)

    d_data = np_data[:, :down_time_len * down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len * down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))

    d_data = d_data.transpose()

    # print('after downsample', d_data.shape, d_labels.shape)

    return d_data.tolist(), d_labels.tolist()


def trim_and_dedup_columns(columns, prefix_len=46):
    def _trim(col):
        col = str(col).strip()
        return col[prefix_len:] if len(col) > prefix_len else col

    trimmed = [_trim(col) for col in columns]
    counts = {}
    deduped = []
    for col in trimmed:
        base = col or 'col'
        count = counts.get(base, 0)
        deduped.append(f'{base}__{count}' if count else base)
        counts[base] = count + 1
    return deduped


def main():
    # 1. 读数据
    train = pd.read_csv('./data/WADI/WADI_14days_new.csv', index_col=0, low_memory=False)
    # test 第一行是数字表头，第二行才是真正名字
    test = pd.read_csv('./data/WADI/WADI_attackdataLABLE.csv', header=1, low_memory=False)

    # 如果 train 的前 3 列不是特征（比如行号、时间），按原逻辑裁剪
    train = train.iloc[:, 3:]

    # 2. 先把列名清洗下（去掉两边空格）
    train = train.rename(columns=lambda x: str(x).strip())
    test = test.rename(columns=lambda x: str(x).strip())

    # 3. 在 test 中找到标签列（带有 Attack LABLE 关键字的）
    attack_cols = [c for c in test.columns if 'Attack LABLE' in c]
    if len(attack_cols) != 1:
        raise ValueError(f'期望只找到 1 个 Attack LABLE 列，实际找到 {len(attack_cols)} 个: {attack_cols}')
    attack_col = attack_cols[0]

    # 4. 提取标签，并转换成 0/1（1=无攻击, -1=有攻击）
    test_labels = test[attack_col].copy()
    test_labels = test_labels.replace({1: 0, -1: 1})

    # 5. 从 test 的特征中删除标签列
    test = test.drop(columns=[attack_col])

    # 6. 如果 test 的前 3 列也不是特征（比如 Row, Date, Time），此时再统一裁剪
    # 注意：train 已经是从第 4 列开始了，这里 test 也从第 4 列开始，保证特征列对齐
    # 由于 test 比 train 多 1 列（PLANT_START_STOP_LOG/TOTAL_CONS_REQUIRED_FLOW 等），
    # 这里通过与 train 做列交集来对齐特征空间
    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    # 7. 数值化 + 填充缺失
    train = train.apply(pd.to_numeric, errors='coerce')
    test = test.apply(pd.to_numeric, errors='coerce')

    train = train.fillna(train.mean(numeric_only=True))
    test = test.fillna(test.mean(numeric_only=True))

    train = train.fillna(0)
    test = test.fillna(0)

    # 8. 构造 train_labels（全 0，因为 train 是正常数据）
    train_labels = np.zeros(len(train))

    # 9. 去掉列名前缀并去重
    clean_cols = trim_and_dedup_columns(train.columns)
    train.columns = clean_cols
    test.columns = clean_cols

    # 10. 归一化
    x_train, x_test = norm(train.values, test.values)

    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]

    # 11. 下采样
    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns=train.columns)
    test_df = pd.DataFrame(d_test_x, columns=test.columns)

    test_df['attack'] = d_test_labels
    # 如果需要 train 的标签可以打开这行
    # train_df['attack'] = d_train_labels

    # 保留你原来的截断逻辑
    train_df = train_df.iloc[2160:]

    train_df.to_csv('./data/WADI/processing/WADI_train.csv', index=False)
    test_df.to_csv('./data/WADI/processing/WADI_test.csv', index=False)


if __name__ == '__main__':
    main()
