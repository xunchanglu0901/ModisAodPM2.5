# -*- coding: utf-8 -*-
# 作者: xcl
# 时间: 2019/10/10 0:07

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
import keras
from keras.layers import Dense, core
from keras.layers import LSTM
import pandas as pd
import os, copy, random
from keras.models import Sequential, load_model
# 读取
df1 = pd.read_excel('d:\\PM2018_all+AODS.xlsx',index_col='日期')
data_ts_df = df1[['tm_mon', 'id']]
for ccc in data_ts_df.columns:
    data_ts_df[ccc] = data_ts_df[ccc].map(lambda x: str(x))
# 虚拟变量
data_get_dummies1 = pd.get_dummies(data_ts_df[['tm_mon']], drop_first=True)
data_get_dummies3 = pd.get_dummies(data_ts_df[['id']], drop_first=True)
data_dummies = pd.concat([data_get_dummies1, data_get_dummies3, data_ts_df[['tm_mon']], data_ts_df[['id']]], axis=1)

# 去掉无用列
data_to_std = df1.drop(['tm_mon', 'id' ], axis=1)
# 标准化
data_std = copy.deepcopy(data_to_std)
mean_pm = data_std['PM25'].mean()
std_pm = data_std['PM25'].std()

for col in data_std:
    mean = data_std[col].mean()
    std = data_std[col].std()
    data_std[col] = data_std[col].map(lambda x:(x-mean)/std)

# 标准化后的数据矩阵
data_out = pd.concat([data_dummies, data_std], join='outer', axis=1)
# 标准化前的数据矩阵
data_out2 = pd.concat([data_dummies, data_to_std], join='outer', axis=1)  # 标准化前的真实值

MAE_list = []
RE_list = []
MSE_list = []
for t_numb in range(0, 5):
    # 划分
    idlist = list(range(1, 153))
    slice1 = random.sample(idlist, 38)  # 从list中随机获取5个元素，作为一个片断返回
    slice2 = []
    for idx in idlist:
        if idx not in slice1:
            idx = str(idx)
            slice2.append(idx)
    slice1 = [str(j) for j in slice1]
    data_test2 = data_out2[data_out2['id'].isin(slice1)]
    # print(data_test2.PM25)  # 这才是真实值

    # 划分标准化后的训练集测试集, 用于训练
    data_test = data_out[data_out['id'].isin(slice1)]
    data_train = data_out[data_out['id'].isin(slice2)]

    input_list = ['AOD_0','AOD_1', 'AOD_2', 'AOD_3', 'AOD_4', 'AOD_5', 'AOD_6', 'AOD_7', 'AOD_8', 'AOD_9',
                     'AOD_10', 'AOD_11', 'AOD_12', 'AOD_13', 'AOD_14', 'AOD_15', 'AOD_16']
    for c1 in data_get_dummies1.columns:
        input_list.append(c1)
    for c3 in data_get_dummies3.columns:
        input_list.append(c3)

    # 生成虚拟训练数据
    x_test = data_test[input_list].values.reshape(38,365,len(input_list)) # pm+id ==2
    x_train = data_train[input_list].values.reshape(114,365,len(input_list)) # pm+id ==2


    # 生成虚拟验证数据
    y_test = data_test[['PM25']].values.reshape(38,365) # pm+id ==2 # 不要（，，1）
    y_train = data_train[['PM25']].values.reshape(114,365) # pm+id ==2

    # 真实数据
    y_test2 = data_test2[['PM25']].values.reshape(38,365) # pm+id ==2 # 不要（，，1）



    timesteps = 365
    data_dim = len(input_list)

    # 期望输入数据尺寸: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))

    model.add(LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
    model.add(core.Dropout(rate=0.01))
    model.add(LSTM(32,))  # 返回维度为 32 的单个向量
    model.add(Dense(365, activation=keras.layers.LeakyReLU(alpha=0.2),kernel_regularizer=keras.regularizers.l2(0.01)))

    model.compile(loss=['mean_absolute_error'],
                  optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=114, epochs=20,verbose=2) # 114 114个实验站
    res=model.predict(x_test)


    list1 = []
    list2 =[]
    list3 =[]

    for j in range(0,38):
        for i in range(0,365):
            # print(i)
            a = res[j][i]*std_pm+mean_pm-y_test2[j][i]
            a = abs(a)
            list1.append(a)
            b = a/abs(y_test2[j][i])
            list2.append(b)
            c = a**2
            list3.append(c)

    MAE = np.average(list1)
    RE = np.average(list2)
    MSE = np.average(list3)
    print('第%s次实验, mae:' % t_numb, MAE)
    print('第%s次实验, re:' % t_numb, RE)
    print('第%s次实验, mse:' % t_numb, MSE)
    MSE_list.append(MSE)
    RE_list.append(RE)
    MAE_list.append(MAE)
print('mae', np.average(MAE_list))
print('re', np.average(RE_list))
print('mse', np.average(MSE_list))

a = []
a.append(MAE_list)
a.append(RE_list)
a.append(MSE_list)

a = pd.DataFrame(a)
a.to_excel('lstm100_sta.xlsx')
# os.system('shutdown -s -f -t 60')
