# -*- coding: utf-8 -*-
# 作者：xcl
# 时间：2019/6/20  23:52
# -*- coding: utf-8 -*-
# 作者：xcl
# 时间：2019/6/20  21:27
from keras import layers, Input
import datetime  # 程序耗时
import random
import pandas as pd
import keras
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, core, add
from keras.models import Model
import copy
import numpy as np
from sklearn.utils import shuffle
import os


# 开始计算耗时
start_time = datetime.datetime.now()
# 读取
input_path = 'D:\\雨雪+2018_new_pm_aod_interpolate.xlsx'
data_all = pd.read_excel(input_path, index_col='日期')
# 去空
data_all = data_all.dropna()
data_ts_df = data_all[['tm_mon', 'tm_mday',
                       'tm_wday', 'tm_yday', 'tm_week', 'id']]
# 虚拟变量
for ccc in data_ts_df.columns:
    data_ts_df[ccc] = data_ts_df[ccc].map(lambda x: str(x))
data_get_dummies1 = pd.get_dummies(data_ts_df[['tm_mon']], drop_first=True)
data_get_dummies3 = pd.get_dummies(data_ts_df[['id']], drop_first=True)
data_dummies = pd.concat([data_get_dummies1,
                          data_get_dummies3,
                          data_ts_df[['tm_mon']],
                          data_ts_df[['id']]],
                         axis=1)

# 去掉不标准化列
data_to_std = data_all.drop(
    ['tm_mon', 'tm_mday', 'tm_wday', 'tm_yday', 'tm_week','id' ], axis=1)


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


# 打乱
# data_all = shuffle(data_all, random_state=1027)
# 耗时
time_list = []
# 误差
MAE_list = []
RE_list = []
MSE_list = []
for t_numb in range(0, 10):
    # 划分
    idlist = list(range(1, 13))
    slice1 = random.sample(idlist, 3)  # 从list中随机获取5个元素，作为一个片断返回
    slice2 = []
    for idx in idlist:
        if idx not in slice1:
            idx = str(idx)
            slice2.append(idx)
    slice1 = [str(j) for j in slice1]
    # 划分不标准化下的训练集测试集, 用于检验
    data_test2 = data_out2[data_out2["tm_mon"].isin(slice1)]
    # print(data_test2.PM25)  # 这才是真实值

    # 划分标准化后的训练集测试集, 用于训练
    data_test = data_out[data_out["tm_mon"].isin(slice1)]
    data_train = data_out[data_out["tm_mon"].isin(slice2)]
    # print(data_train.index)

    # AOD
    data_aod_test = data_test[['AOD_0']]
    data_aod_train = data_train[['AOD_0']]

    # 气象
    data_sky_test = data_test[[
        'cloudCover',
        'dewPoint',
        'humidity',
        'sunTime',
        'visibility',
        'windGust',
        'windSpeed',
        'temperature',
        'tempMM', 'tempHL', 'atempMM', 'atempHL',
        'pressure',
        'precipIntensity',
        'precipAccumulation']]

    data_sky_train = data_train[[
        'cloudCover',
        'dewPoint',
        'humidity',
        'sunTime',
        'visibility',
        'windGust',
        'windSpeed',
        'temperature',
        'tempMM', 'tempHL', 'atempMM', 'atempHL',
        'pressure',
        'precipIntensity',
        'precipAccumulation',
    ]]

    # 时间特征
    data_time_test = data_test[['tm_mon_10',
                                'tm_mon_11',
                                'tm_mon_12',
                                'tm_mon_2',
                                'tm_mon_3',
                                'tm_mon_4',
                                'tm_mon_5',
                                'tm_mon_6',
                                'tm_mon_7',
                                'tm_mon_8',
                                'tm_mon_9',
                                ]]

    data_time_train = data_train[['tm_mon_10',
                                  'tm_mon_11',
                                  'tm_mon_12',
                                  'tm_mon_2',
                                  'tm_mon_3',
                                  'tm_mon_4',
                                  'tm_mon_5',
                                  'tm_mon_6',
                                  'tm_mon_7',
                                  'tm_mon_8',
                                  'tm_mon_9',
                                  ]]
    # 空间特征
    data_station_test = data_test[data_get_dummies3.columns]

    data_station_train = data_train[data_get_dummies3.columns]
    # 时滞
    data_t1_test = data_test[['AOD_0_T1',
                              'cloudCover_T1',
                              'dewPoint_T1',
                              'humidity_T1',
                              'sunTime_T1',
                              'visibility_T1',
                              'windSpeed_T1',
                              'temperature_T1',
                              'pressure_T1',
                              'precipIntensity_T1',
                              'precipAccumulation_T1',
                              ]]
    """
    data_t1_train = data_train[['AOD_0_T1',
                                'apparentTemperatureHigh_T1',
                                'apparentTemperatureLow_T1',
                                'apparentTemperatureMax_T1',
                                'apparentTemperatureMin_T1',
                                'cloudCover_T1',
                                'dewPoint_T1',
                                'humidity_T1',
                                'sunTime_T1',
                                'temperatureHigh_T1',
                                'temperatureLow_T1',
                                'temperatureMax_T1',
                                'temperatureMin_T1',
                                'visibility_T1',
                                'windBearing_T1',
                                'windGust_T1',
                                'windSpeed_T1',
                                'apparentTemperature_T1',
                                'temperature_T1',                          'pressure_T1',
                              'precipIntensity_T1',
                              'precipIntensityMax_T1',
                              'precipAccumulation_T1',]]
                              """
    data_t1_train = data_train[['AOD_0_T1',
                                'cloudCover_T1',
                                'dewPoint_T1',
                                'humidity_T1',
                                'sunTime_T1',
                                'visibility_T1',
                                'windSpeed_T1',
                                'temperature_T1',
                                'pressure_T1',
                                'precipIntensity_T1',
                                'precipAccumulation_T1',
                                ]]
    # NDVI
    data_ndvi_test = data_test[['NDVI_0']]
    data_ndvi_train = data_train[['NDVI_0']]

    # AODS
    data_aods_test = data_test[['AOD_1',
                                'AOD_2',
                                'AOD_3',
                                'AOD_4',
                                'AOD_5',
                                'AOD_6',
                                'AOD_7',
                                'AOD_8',
                                'AOD_9',
                                'AOD_10',
                                'AOD_11',
                                'AOD_12',
                                'AOD_13',
                                'AOD_14',
                                'AOD_15',
                                'AOD_16']]
    data_aods_train = data_train[['AOD_1',
                                  'AOD_2',
                                  'AOD_3',
                                  'AOD_4',
                                  'AOD_5',
                                  'AOD_6',
                                  'AOD_7',
                                  'AOD_8',
                                  'AOD_9',
                                  'AOD_10',
                                  'AOD_11',
                                  'AOD_12',
                                  'AOD_13',
                                  'AOD_14',
                                  'AOD_15',
                                  'AOD_16']]
    # 污染物
    data_pm_test = data_test[['PM25']]
    data_pm_train = data_train[['PM25']]

    # 输入1和2的变量数,维度 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!是否可以不同维度
    inputAOD = Input(shape=(len(data_aod_test.columns),))
    inputSky = Input(shape=(len(data_sky_test.columns),))
    inputTime = Input(shape=(len(data_time_test.columns),))
    inputStaion = Input(shape=(len(data_station_test.columns),))
    inputT1 = Input(shape=(len(data_t1_test.columns),))
    inputNDVI = Input(shape=(len(data_ndvi_test.columns),))
    inputAODs = Input(shape=(len(data_aods_test.columns),))

    # 输入1
    x1 = layers.Dense(
        24, activation=keras.layers.advanced_activations.LeakyReLU(
            alpha=0.5))(inputAOD)
    x1_residual_connection1 = Dense(24,
                                      activation=keras.layers.LeakyReLU(alpha=0.2))(x1)
    x1_residual_connection2 = Dense(
        24, activation=keras.layers.advanced_activations.ELU(alpha=1.0))(x1_residual_connection1)
    x1_residual_output = add([x1, x1_residual_connection2])
    x1 = layers.Dense(12, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(x1_residual_output)
    x1= core.Dropout(rate=0.01)(x1)
    x1 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(x1)
    x1 = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(x1)
    x1 = Model(inputs=inputAOD, outputs=x1)
    # 输入2
    x2 = layers.Dense(
        24, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(inputSky)
    x2_residual_connection1 = Dense(24,
                                    activation=keras.layers.LeakyReLU(alpha=0.2))(x2)
    x2_residual_connection2 = Dense(24, activation=keras.layers.advanced_activations.ELU(alpha=1.0))(x2_residual_connection1)
    x2_residual_output = add([x2, x2_residual_connection2])
    x2 = layers.Dense(12, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(x2_residual_output)
    x2 = core.Dropout(rate=0.01)(x2)
    x2 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(x2)
    x2 = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(x2)
    x2 = Model(inputs=inputSky, outputs=x2)

    # 输入3
    x3 = layers.Dense(
        24, activation=keras.layers.advanced_activations.LeakyReLU(
            alpha=0.5))(inputTime)
    x3_residual_connection1 = Dense(24, activation=keras.layers.LeakyReLU(alpha=0.2))(x3)
    x3_residual_connection2 = Dense(24, activation=keras.layers.advanced_activations.ELU(alpha=1.0))(x3_residual_connection1)
    x3_residual_output = add([x3, x3_residual_connection2])
    x3 = layers.Dense(12, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(x3_residual_output)
    x3 = core.Dropout(rate=0.01)(x3)
    x3 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(x3)
    x3 = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(x3)
    x3 = Model(inputs=inputTime, outputs=x3)

    # 输入4
    x4 = layers.Dense(24, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(inputStaion)
    x4_residual_connection1 = Dense(24, activation=keras.layers.LeakyReLU(alpha=0.2))(x4)
    x4_residual_connection2 = Dense(24, activation=keras.layers.advanced_activations.ELU(alpha=1.0))(x4_residual_connection1)
    x4_residual_output = add([x4, x4_residual_connection2])
    x4 = layers.Dense(12, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(x4_residual_output)
    x4 = core.Dropout(rate=0.01)(x4)
    x4 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(x4)
    x4 = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(x4)
    x4 = Model(inputs=inputStaion, outputs=x4)
    # 输入5
    x5 = layers.Dense(24, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(inputT1)
    x5_residual_connection1 = Dense(24, activation=keras.layers.LeakyReLU(alpha=0.2))(x5)
    x5_residual_connection2 = Dense(24, activation=keras.layers.advanced_activations.ELU(alpha=1.0))(x5_residual_connection1)
    x5_residual_output = add([x5, x5_residual_connection2])
    x5 = layers.Dense(12, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(x5_residual_output)
    x5 = core.Dropout(rate=0.01)(x5)
    x5 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(x5)
    x5 = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(x5)
    x5 = Model(inputs=inputT1, outputs=x5)
    # 输入6
    x6 = layers.Dense(24, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(inputNDVI)
    x6_residual_connection1 = Dense(24, activation=keras.layers.LeakyReLU(alpha=0.2))(x6)
    x6_residual_connection2 = Dense(24, activation=keras.layers.advanced_activations.ELU(alpha=1.0))(x6_residual_connection1)
    x6_residual_output = add([x6, x6_residual_connection2])
    x6 = layers.Dense(12, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(x6_residual_output)
    x6 = core.Dropout(rate=0.01)(x6)
    x6 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(x6)
    x6 = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(x6)
    x6 = Model(inputs=inputNDVI, outputs=x6)
    # 输入7
    x7 = layers.Dense(24, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(inputAODs)
    x7_residual_connection1 = Dense(24, activation=keras.layers.LeakyReLU(alpha=0.2))(x7)
    x7_residual_connection2 = Dense(24, activation=keras.layers.advanced_activations.ELU(alpha=1.0))(x7_residual_connection1)
    x7_residual_output = add([x7, x7_residual_connection2])
    x7 = layers.Dense(12, activation=keras.layers.advanced_activations.LeakyReLU(alpha=0.5))(x7_residual_output)
    x7 = core.Dropout(rate=0.01)(x7)
    x7 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(x7)
    x7 = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(x7)
    x7 = Model(inputs=inputAODs, outputs=x7)

    combined = layers.concatenate(
        [x1.output, x2.output, x3.output, x4.output, x5.output, x6.output, x7.output])
    # 输出层
    # 全连接层 1
    res_x1 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(combined)

    # 残差连接层
    res_residual_connection1 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(res_x1)

    res_residual_connection2 = Dense(
        8, activation=keras.layers.LeakyReLU(alpha=0.2))(res_residual_connection1)

    res_residual_output = add([res_x1, res_residual_connection2])

    # 全连接层 2
    res_x2 = Dense(8, activation=keras.layers.LeakyReLU(alpha=0.2))(res_residual_output)
    res_x3 = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(res_x2)
    # Dropout
    res_y = core.Dropout(rate=0.01)(res_x3)
    z = Dense(4, activation=keras.layers.LeakyReLU(alpha=0.2))(res_y)
    z = Dense(1, activation=keras.layers.LeakyReLU(alpha=0.2),kernel_regularizer=keras.regularizers.l2(0.01))(z)
    # 建立模型
    model = Model(
        inputs=[
            x1.input,
            x2.input,
            x3.input,
            x4.input,
            x5.input,
            x6.input,
            x7.input],
        outputs=z)
    # 模型编译
    model.compile(
        loss=['mean_absolute_error'],
        # optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.00001),
        # optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
        # optimizer=keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.00001),
        # optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
        optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
        # epsilon=None, decay=0.0, amsgrad=False),
        metrics=["accuracy"])
    # 计算耗时
    starttime = datetime.datetime.now().second
    # 运行
    model.fit([
        data_aod_train,
        data_sky_train,
        data_time_train,
        data_station_train,
        data_t1_train,
        data_ndvi_train,
        data_aods_train
    ],
        data_pm_train,
        epochs=20,
        batch_size=5120)
    # 耗时
    endtime = datetime.datetime.now().second
    t_gap = endtime - starttime
    time_list.append(t_gap)

    res = model.predict([data_aod_test,
                         data_sky_test,
                         data_time_test,
                         data_station_test,
                         data_t1_test,
                         data_ndvi_test,
                         data_aods_test])
    datares = res - data_pm_test
    datares.PM25 = datares.PM25.map(lambda x: abs(x))
    data_predt = pd.concat([datares, data_pm_test], axis=1)
    data_predt.columns = ["差值", '真']
    data_predt['差值'] = data_predt['差值'].map(lambda x: abs(x))
    data_predt['百分误'] = data_predt['差值'].div(data_predt["真"])
    data_predt['差值2'] = data_predt['差值'].map(lambda x: x**2)
    e_AME = np.average(data_predt['差值'])
    e_RE = np.average(data_predt['百分误'])
    e_MSE = np.average(data_predt['差值2'])
    # 还原，反标准化
    res2 = [float((j * std_pm) + mean_pm) for j in res]
    res2 = pd.DataFrame(res2, index=data_pm_test.index, columns=['PM25'])

    datares = res2 - data_test2[['PM25']]  # 预测-真实
    # print(datares)
    datares.PM25 = datares.PM25.map(lambda x: abs(x))
    data_predt = pd.concat([datares, data_test2.PM25], axis=1)  # 标准化后真值变化了 需要修改

    data_predt.columns = ["差值", '真']
    data_predt['差值'] = data_predt['差值'].map(lambda x: abs(x))
    data_predt['百分误'] = data_predt['差值'].div(data_predt["真"])
    data_predt['差值2'] = data_predt['差值'].map(lambda x: x ** 2)

    MAE = np.average(data_predt['差值'])
    RE = np.average(data_predt['百分误'])
    MSE = np.average(data_predt['差值2'])

    print('第%s次实验, mae:' % t_numb, np.average(data_predt['差值']))
    print('第%s次实验, re:' % t_numb, np.average(data_predt['百分误']))
    print('第%s次实验, mse:' % t_numb, np.average(data_predt['差值2']))

    MSE_list.append(MSE)
    RE_list.append(RE)
    MAE_list.append(MAE)
    print('=========================== %s ===========================' % t_numb)
print('mae', np.average(MAE_list))
print('re', np.average(RE_list))
print('mse', np.average(MSE_list))

a = []
a.append(MAE_list)
a.append(RE_list)
a.append(MSE_list)

a = pd.DataFrame(a)
a.to_excel('多输入_随机月份_标准化.xlsx')
# os.system('shutdown -s -f -t 60')

print('平均耗时', np.average(time_list))
print('总耗时', np.sum(time_list))