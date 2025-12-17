# import numpy as np
# import pandas as pd
#
# normal = pd.read_csv("../data/SWaT/swat_train2.csv")
# attack = pd.read_csv("../data/SWaT/swat2.csv",sep=";")
#
# normal['Timestamp'] = pd.to_datetime(normal['Timestamp'])
# del normal['Normal/Attack']
#
# normal = normal.rename(columns={'Timestamp':'datetime'})
#
# datetime = normal['datetime']
# del normal['datetime']
#
# for i in list(normal):
#     normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
# normal = normal.astype(float)
# normal['datetime']= datetime
#
# normal.to_csv('../data/SWaT/processing/SWaT_train.csv', index=False)
#
# attack['Timestamp'] = pd.to_datetime(attack['Timestamp'])
# attack = attack.rename(columns={'Timestamp':'datetime'})
# datetime = attack['datetime']
# del attack['datetime']
#
# labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
# del attack['Normal/Attack']
#
# for i in list(attack):
#     attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
# attack = attack.astype(float)
#
# attack['datetime'] = datetime
# attack['label'] = labels
#
# attack.to_csv('../data/SWaT/processing/SWaT_test.csv', index=False)

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


# max min(0-1)
def norm(train, test):

    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train) # scale training data to [0,1] range
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

    d_data = np_data[:, :down_time_len*down_len].reshape(col_num, -1, down_len)
    d_data = np.median(d_data, axis=2).reshape(col_num, -1)

    d_labels = np_labels[:down_time_len*down_len].reshape(-1, down_len)
    # if exist anomalies, then this sample is abnormal
    d_labels = np.round(np.max(d_labels, axis=1))


    d_data = d_data.transpose()

    return d_data.tolist(), d_labels.tolist()


def main():

    test = pd.read_csv('./data/SWaT/swat_train2.csv', index_col=0)
    train = pd.read_csv('./data/SWaT/swat2.csv', index_col=0)


    test = test.iloc[:, :]
    train = train.iloc[:, :]

    train = train.fillna(train.mean())
    test = test.fillna(test.mean())
    train = train.fillna(0)
    test = test.fillna(0)

    # trim column names
    train = train.rename(columns=lambda x: x.strip())
    test = test.rename(columns=lambda x: x.strip())

    # print(len(test.columns),test.columns)
    # print(len(train.columns),train.columns)

    test = test.rename(columns={'Normal / Attack': 'attack'})


    train_labels = train.attack
    test_labels = test.attack

    train = train.drop(columns=['attack'])
    test = test.drop(columns=['attack'])


    x_train, x_test = norm(train.values, test.values)


    for i, col in enumerate(train.columns):
        train.loc[:, col] = x_train[:, i]
        test.loc[:, col] = x_test[:, i]


    d_train_x, d_train_labels = downsample(train.values, train_labels, 10)
    d_test_x, d_test_labels = downsample(test.values, test_labels, 10)

    train_df = pd.DataFrame(d_train_x, columns = train.columns)
    test_df = pd.DataFrame(d_test_x, columns = test.columns)

    test_df['attack'] = d_test_labels
    train_df['attack'] = d_train_labels

    train_df = train_df.iloc[2160:]

    # print(train_df.values.shape)
    # print(test_df.values.shape)

    train_df.to_csv('./data/SWaT/processing/SWaT_train.csv')
    test_df.to_csv('./data/SWaT/processing/SWaT_test.csv')

    f = open('./data/SWaT/processing/list.txt', 'w')
    for col in train.columns:
        f.write(col+'\n')
    f.close()

if __name__ == '__main__':
    main()