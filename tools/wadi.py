import pandas as pd

train_new = pd.read_csv('../data/WADI/WADI_14days_new.csv')
test_new = pd.read_csv('../data/WADI/WADI_attackdataLABLE.csv', skiprows=1)

test = pd.read_csv('../data/WADI/WADI_attackdata.csv')
train = pd.read_csv('../data/WADI/WADI_14days.csv', skiprows=4)

def recover_date(str1, str2):
    return str1+" "+str2
train["datetime"] = train.apply(lambda x : recover_date(x['Date'], x['Time']), axis=1)
train["datetime"] = pd.to_datetime(train['datetime'])

train_time = train[['Row', 'datetime']]
train_new_time = pd.merge(train_new, train_time, how='left', on='Row')
del train_new_time['Row']
del train_new_time['Date']
del train_new_time['Time']
train_new_time.to_csv('../data/WADI/processing/WADI_train.csv', index=False)

test["datetime"] = test.apply(lambda x : recover_date(x['Date'], x['Time']), axis=1)
test["datetime"] = pd.to_datetime(test['datetime'])
test = test.loc[-2:, :]
test_new = test_new.rename(columns={'Row ':'Row'})

test_time = test[['Row', 'datetime']]
test_new_time = pd.merge(test_new, test_time, how='left', on='Row')

del test_new_time['Row']
del test_new_time['Date ']
del test_new_time['Time']

test_new_time = test_new_time.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)':'label'})
test_new_time.loc[test_new_time['label'] == 1, 'label'] = 0
test_new_time.loc[test_new_time['label'] == -1, 'label'] = 1

test_new_time.to_csv('../data/WADI/processing/WADI_test.csv', index=False)