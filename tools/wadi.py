import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# train_new = pd.read_csv('../data/WADI/WADI_14days_new.csv')
# test_new = pd.read_csv('../data/WADI/WADI_attackdataLABLE.csv', skiprows=1)
#
# test = pd.read_csv('../data/WADI/WADI_attackdata.csv')
# train = pd.read_csv('../data/WADI/WADI_14days.csv', skiprows=4)
#
# def recover_date(str1, str2):
#     return str1+" "+str2
# train["datetime"] = train.apply(lambda x : recover_date(x['Date'], x['Time']), axis=1)
# train["datetime"] = pd.to_datetime(train['datetime'])
#
# train_time = train[['Row', 'datetime']]
# train_new_time = pd.merge(train_new, train_time, how='left', on='Row')
# del train_new_time['Row']
# del train_new_time['Date']
# del train_new_time['Time']
# train_new_time.to_csv('../data/WADI/processing/WADI_train.csv', index=False)
#
# test["datetime"] = test.apply(lambda x : recover_date(x['Date'], x['Time']), axis=1)
# test["datetime"] = pd.to_datetime(test['datetime'])
# test = test.loc[-2:, :]
# test_new = test_new.rename(columns={'Row ':'Row'})
#
# test_time = test[['Row', 'datetime']]
# test_new_time = pd.merge(test_new, test_time, how='left', on='Row')
#
# del test_new_time['Row']
# del test_new_time['Date ']
# del test_new_time['Time']
#
# test_new_time = test_new_time.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)':'label'})
# test_new_time.loc[test_new_time['label'] == 1, 'label'] = 0
# test_new_time.loc[test_new_time['label'] == -1, 'label'] = 1
#
# test_new_time.to_csv('../data/WADI/processing/WADI_test.csv', index=False)
# import numpy as np
# import pandas as pd
# import re
# from sklearn.preprocessing import MinMaxScaler
#
#
# # max min(0-1)
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


def main():
    # 读入时先关掉 low_memory，避免类型被分块推断
    train = pd.read_csv('./data/WADI/WADI_14days_new.csv', index_col=0, low_memory=False)
    test = pd.read_csv('./data/WADI/WADI_attackdataLABLE.csv', index_col=0, low_memory=False)

    train = train.iloc[:, 2:]
    test = test.iloc[:, 2:]

    # 1. 先把所有列尽量转成数值类型，无法转换的变成 NaN
    train = train.apply(pd.to_numeric, errors='coerce')
    test = test.apply(pd.to_numeric, errors='coerce')

    # 2. 再用各列均值填充 NaN
    train = train.fillna(train.mean(numeric_only=True))
    test = test.fillna(test.mean(numeric_only=True))

    # 3. 再兜底把还剩下的 NaN 填成 0（如果有）
    train = train.fillna(0)
    test = test.fillna(0)

    # 后面保持不变
    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    train_labels = np.zeros(len(train))

    test = test.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': 'attack'})

    test_labels = test.attack

    # train = train.drop(columns=['attack'])

    test = test.drop(columns=['attack'])

    cols = [x[46:] for x in train.columns]  # remove column name prefixes
    train.columns = cols
    test.columns = cols

    x_train, x_test = norm(train.values, test.values)

    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]

    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns=train.columns)
    test_df = pd.DataFrame(d_test_x, columns=test.columns)

    test_df['attack'] = d_test_labels
    # train_df['attack'] = d_train_labels

    train_df = train_df.iloc[2160:]

    train_df.to_csv('./data/WADI/processing/WADI_train.csv')
    test_df.to_csv('./data/WADI/processing/WADI_test.csv')

    # f = open('./list.txt', 'w')
    # for col in train.columns:
    #     f.write(col + '\n')
    # f.close()


if __name__ == '__main__':
    main()
