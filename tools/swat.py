import numpy as np
import pandas as pd

normal = pd.read_csv("../data/SWaT/swat_train2.csv")
attack = pd.read_csv("../data/SWaT/swat2.csv",sep=";")

normal['Timestamp'] = pd.to_datetime(normal['Timestamp'])
del normal['Normal/Attack']

normal = normal.rename(columns={'Timestamp':'datetime'})

datetime = normal['datetime']
del normal['datetime']

for i in list(normal):
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)
normal['datetime']= datetime

normal.to_csv('../data/SWaT/processing/SWaT_train.csv', index=False)

attack['Timestamp'] = pd.to_datetime(attack['Timestamp'])
attack = attack.rename(columns={'Timestamp':'datetime'})
datetime = attack['datetime']
del attack['datetime']

labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
del attack['Normal/Attack']

for i in list(attack):
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)

attack['datetime'] = datetime
attack['label'] = labels

attack.to_csv('../data/SWaT/processing/SWaT_test.csv', index=False)