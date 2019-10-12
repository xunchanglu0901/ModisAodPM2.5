# -*- coding: utf-8 -*-
# 作者: xcl
# 时间: 2019/9/21 23:54


# 库
from random import choice
import random
from sklearn.ensemble import AdaBoostRegressor
from keras.models import Sequential, Model
from keras import layers, Input
import keras
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import KFold,StratifiedKFold
import datetime  # 程序耗时

import pandas as pd
import keras
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, core, add
from keras.models import Model
import os
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


input_path = 'D:\\雨雪+2018_new_pm_aod_interpolate.xlsx'

data_all = pd.read_excel(input_path, index_col='日期')

"""
del data_all['pressure_T1']
del data_all['pressure']
"""
data_all = data_all.dropna()
data_ts_df = data_all[['tm_mon', 'tm_mday',
                       'tm_wday', 'tm_yday', 'tm_week', 'id']]
# 虚拟变量
for ccc in data_ts_df.columns:
    data_ts_df[ccc] = data_ts_df[ccc].map(lambda x: str(x))
data_get_dummies1 = pd.get_dummies(data_ts_df[['tm_mon']], drop_first=True)
data_get_dummies2 = pd.get_dummies(data_ts_df[['tm_mday']], drop_first=True)
data_get_dummies3 = pd.get_dummies(data_ts_df[['id']], drop_first=True)
data_dummies = pd.concat([data_get_dummies1,
                          data_get_dummies2,
                          data_get_dummies3,
                          data_ts_df[['tm_mon']]],
                         axis=1)
list1 = []
for ccc in data_dummies.columns:
    # print(ccc)
    if ccc != 'tm_mon':
        list1.append(ccc)

# 去掉无用列; 去掉不需要标准化的列
data_to_std = data_all.drop(
    ['tm_mon', 'tm_mday', 'tm_wday', 'tm_yday', 'tm_week', 'id'], axis=1)

# 不标准化
data_out = pd.concat([data_dummies, data_to_std], join='outer', axis=1)
# 打乱
# data_all = shuffle(data_all, random_state=1027)

MAE_list = []
RE_list = []
MSE_list = []
for t_numb in range(0,20):

    # 划分
    idlist = list(range(1,153))
    slice1 = random.sample(idlist, 38)  #从list中随机获取5个元素，作为一个片断返回
    slice2 = []
    for idx in idlist:
        if idx not in slice1:
            idx = str(idx)
            slice2.append(idx)
    slice1 = [str(j) for j in slice1]

    data_test = data_out[data_out["id"].isin(slice1)]
    data_train = data_out[data_out["id"].isin(slice2)]
    # AOD
    data_aod_test = data_test[['AOD_0']]
    data_aod_train = data_train[['AOD_0']]


    # 气象
    data_sky_test = data_test[[
                               'cloudCover',
                               'dewPoint',
                               'humidity',

                               'sunTime',
    'tempMM','tempHL','atempMM','atempHL',

                               'visibility',
                               'windGust',
                               'windSpeed',
                               'apparentTemperature',
                               'temperature',

                               'pressure',
                               'precipIntensity',
                               'precipAccumulation']]

    data_sky_train = data_train[[
                                 'cloudCover',
                                 'dewPoint',
                                 'humidity',

                                 'sunTime',
    'tempMM','tempHL','atempMM','atempHL',


                                 'visibility',
                                 'windGust',
                                 'windSpeed',
                                 'apparentTemperature',
                                 'temperature',

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
    """
    data_ndvi_train = data_train[['tm_mday_10',
                                  'tm_mday_11',
                                  'tm_mday_12',
                                  'tm_mday_13',
                                  'tm_mday_14',
                                  'tm_mday_15',
                                  'tm_mday_16',
                                  'tm_mday_17',
                                  'tm_mday_18',
                                  'tm_mday_19',
                                  'tm_mday_2',
                                  'tm_mday_20',
                                  'tm_mday_21',
                                  'tm_mday_22',
                                  'tm_mday_23',
                                  'tm_mday_24',
                                  'tm_mday_25',
                                  'tm_mday_26',
                                  'tm_mday_27',
                                  'tm_mday_28',
                                  'tm_mday_29',
                                  'tm_mday_3',
                                  'tm_mday_30',
                                  'tm_mday_31',
                                  'tm_mday_4',
                                  'tm_mday_5',
                                  'tm_mday_6',
                                  'tm_mday_7',
                                  'tm_mday_8',
                                  'tm_mday_9']]
    """
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
    # 输入层 全部

    AOD_input = Input(shape=(len(data_aod_test.columns),), name="AOD_input")  # AOD
    AODs_input = Input(shape=(len(data_aods_test.columns),),
                       name="AODs_input")  # AODs

    Meteorology_input = Input(
        shape=(len(data_sky_test.columns),), name="Meteorology_input")  # 气象

    Weather_input = Input(shape=(len(data_t1_test.columns),),
                          name="Weather_input")  # 时滞

    Ndvi_input = Input(shape=(len(data_ndvi_test.columns),),
                       name="NDVI_input")  # NDVI

    Time_input = Input(shape=(len(data_time_test.columns),),
                       name="Time_input")  # 时间

    Station_input = Input(
        shape=(len(data_station_test.columns),), name="Station_input")  # 空间
    # 融合层
    aods_concat = concatenate([AOD_input, AODs_input])  # AOD + AODs

    meteorology_concat = concatenate([Meteorology_input, AOD_input])  # AOD + 气象

    weather_concat = concatenate([Weather_input, AOD_input])  # AOD + 时滞

    ndvi_concat = concatenate([Ndvi_input, AOD_input])  # AOD + NDVI

    time_concat = concatenate([Time_input, AOD_input])  # AOD + 时间
    station_concat = concatenate([Station_input, AOD_input])  # AOD + 空间

    allin_concat = concatenate([Meteorology_input,
                                Weather_input,
                                Ndvi_input,
                                Time_input,
                                Station_input,
                                AODs_input,
                                AOD_input])  # 全
    # 全连接层 1

    # AOD + AODs
    aods_x1 = Dense(24,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FC1_aods")(aods_concat)
    # AOD + 气象
    meteorology_x1 = Dense(24, activation=keras.layers.LeakyReLU(
        alpha=0.2), name="FC1_meteorology")(meteorology_concat)
    # AOD + 时滞
    weather_x1 = Dense(24,
                       activation=keras.layers.LeakyReLU(alpha=0.2),
                       name="FC1_T1")(weather_concat)
    # AOD + NDVI
    ndvi_x1 = Dense(24,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FC1_NDVI")(ndvi_concat)
    # AOD + 时间
    time_x1 = Dense(24,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FC1_Time")(time_concat)
    # AOD + 空间
    station_x1 = Dense(24,
                       activation=keras.layers.LeakyReLU(alpha=0.2),
                       name="FC1_Station")(station_concat)
    # 全部特征
    allin_x1 = Dense(24,
                     activation=keras.layers.LeakyReLU(alpha=0.2),
                     name="FC11_AIA")(allin_concat)

    # 残差层
    # AOD + AODs
    aods_residual_connection1 = Dense(24,
                                      activation=keras.layers.LeakyReLU(alpha=0.2),
                                      name="ResidualConnectionAODs")(aods_x1)
    aods_residual_connection2 = Dense(
        24, activation=keras.layers.advanced_activations.ELU(
            alpha=1.0), name="FullConnectionAOD_RC")(aods_residual_connection1)
    # aods_residual_connection = add([aods_x, aods_residual_connection],
    # name="ResidualConnectionAODs_Add")  # 原先版本
    aods_residual_output = add([aods_x1,
                                aods_residual_connection2],
                               name="ResidualConnectionAODs_Add")
    # AOD + 气象
    meteorology_residual_connection1 = Dense(
        24, activation=keras.layers.LeakyReLU(
            alpha=0.2), name="ResidualConnectionMA")(meteorology_x1)
    meteorology_residual_connection2 = Dense(24, activation=keras.layers.LeakyReLU(
        alpha=0.2), name="FullConnectionMA_RC")(meteorology_residual_connection1)
    meteorology_residual_output = add(
        [meteorology_x1, meteorology_residual_connection2], name="ResidualConnectionMA_Add")
    # AOD + 时滞
    weather_residual_connection1 = Dense(
        24, activation=keras.layers.LeakyReLU(
            alpha=0.2), name="ResidualConnectionWA")(weather_x1)
    weather_residual_connection2 = Dense(24, activation=keras.layers.LeakyReLU(
        alpha=0.2), name="FullConnectionWAForRC")(weather_residual_connection1)
    weather_residual_output = add([weather_x1,
                                   weather_residual_connection2],
                                  name="ResidualConnectionWA_Add")
    # AOD + NDVI
    ndvi_residual_connection1 = Dense(
        24, activation=keras.layers.LeakyReLU(
            alpha=0.2), name="ResidualConnectionNDVI")(ndvi_x1)
    ndvi_residual_connection2 = Dense(24, activation=keras.layers.LeakyReLU(
        alpha=0.2), name="FullConnectionNDVI_RC")(ndvi_residual_connection1)
    ndvi_residual_output = add([ndvi_x1,
                                ndvi_residual_connection2],
                               name="ResidualConnectionNDVI_Add")
    # AOD + 时间
    time_residual_connection1 = Dense(
        24, activation=keras.layers.LeakyReLU(
            alpha=0.2), name="ResidualConnectionTime")(time_x1)
    time_residual_connection2 = Dense(24, activation=keras.layers.LeakyReLU(
        alpha=0.2), name="FullConnectionTime_RC")(time_residual_connection1)
    time_residual_output = add([time_x1,
                                time_residual_connection2],
                               name="ResidualConnectionTime_Add")
    # AOD + 空间
    station_residual_connection1 = Dense(
        24, activation=keras.layers.LeakyReLU(
            alpha=0.2), name="ResidualConnectionStation")(station_x1)
    station_residual_connection2 = Dense(24, activation=keras.layers.LeakyReLU(
        alpha=0.2), name="FullConnectionStation_RC")(station_residual_connection1)
    station_residual_output = add([station_x1,
                                   station_residual_connection2],
                                  name="ResidualConnectionStation_Add")
    # 全部特征
    allin_residual_connection1 = Dense(
        24, activation=keras.layers.LeakyReLU(
            alpha=0.2), name="ResidualConnectionAIA")(allin_x1)
    allin_residual_connection2 = Dense(24, activation=keras.layers.LeakyReLU(
        alpha=0.2), name="FullConnectionAIA_RC")(allin_residual_connection1)
    allin_residual_output = add([allin_x1,
                                 allin_residual_connection2],
                                name="ResidualConnectionAIA_Add")

    # 全连接层 2
    # AOD + AODs
    aods_x2 = Dense(12,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionAODs_2")(aods_residual_output)
    # AOD + 气象
    meteorology_x2 = Dense(12,
                           activation=keras.layers.LeakyReLU(alpha=0.2),
                           name="FullConnectionMA_2")(meteorology_residual_output)
    # AOD + 时滞
    weather_x2 = Dense(12,
                       activation=keras.layers.LeakyReLU(alpha=0.2),
                       name="FullConnectionWA_2")(weather_residual_output)
    # AOD + NDVI
    ndvi_x2 = Dense(12,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionNDVI_2")(ndvi_residual_output)
    # AOD + 时间
    time_x2 = Dense(12,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionTime_2")(time_residual_output)
    # AOD + 空间
    station_x2 = Dense(12,
                       activation=keras.layers.LeakyReLU(alpha=0.2),
                       name="FullConnectionStation_2")(station_residual_output)
    # 全部特征
    allin_x2 = Dense(12,
                     activation=keras.layers.LeakyReLU(alpha=0.2),
                     name="FullConnectionAIA_2")(allin_residual_output)

    # Dropout

    # AOD + AODs
    aods_y = core.Dropout(rate=0.01, name="Aods_Module")(aods_x2)

    # AOD + 气象
    meteorology_y = core.Dropout(rate=0.01, name="Meteorology_Module")(meteorology_x2)

    # AOD + 时滞
    weather_y = core.Dropout(rate=0.01, name="Weather_Module")(weather_x2)

    # AOD + NDVI
    ndvi_y = core.Dropout(rate=0.01, name="NDVI_Module")(ndvi_x2)

    # AOD + 时间
    time_y = core.Dropout(rate=0.01, name="Time_Module")(time_x2)

    # AOD + 空间
    station_y = core.Dropout(rate=0.01, name="Station_Module")(station_x2)

    # 全部特征
    allin_y = core.Dropout(rate=0.01, name="AllIn_Module")(allin_x2)

    # 全连接层 3

    # AOD + AODs
    aods_y2 = Dense(8,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionAODs_3x")(aods_y)

    # AOD + 气象
    meteorology_y2 = Dense(8,
                           activation=keras.layers.LeakyReLU(alpha=0.2),
                           name="FullConnectionMA_3x")(meteorology_y)

    # AOD + 时滞
    weather_y2 = Dense(8,
                       activation=keras.layers.LeakyReLU(alpha=0.2),
                       name="FullConnectionWA_3x")(weather_y)

    # AOD + NDVI
    ndvi_y2 = Dense(8,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionNDVI_3x")(ndvi_y)

    # AOD + 时间
    time_y2 = Dense(8,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionTime_3x")(time_y)


    # AOD + 空间
    station_y2 = Dense(8,
                       activation=keras.layers.LeakyReLU(alpha=0.2),
                       name="FullConnectionStation_3x")(station_y)

    # 全部特征
    allin_y2 = Dense(8,
                     activation=keras.layers.LeakyReLU(alpha=0.2),
                     name="FullConnectionAIA_3x")(allin_y)

    # 全连接层 3

    # AOD + AODs
    aods_y2 = Dense(4,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionAODs_3")(aods_y2)

    # AOD + 气象
    meteorology_y2 = Dense(4,
                           activation=keras.layers.LeakyReLU(alpha=0.2),
                           name="FullConnectionMA_3")(meteorology_y2)

    # AOD + 时滞
    weather_y2 = Dense(4,
                       activation=keras.layers.LeakyReLU(alpha=0.2),
                       name="FullConnectionWA_3")(weather_y2)

    # AOD + NDVI
    ndvi_y2 = Dense(4,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionNDVI_3")(ndvi_y2)

    # AOD + 时间
    time_y2 = Dense(4,
                    activation=keras.layers.LeakyReLU(alpha=0.2),
                    name="FullConnectionTime_3")(time_y2)


    # AOD + 空间
    station_y2 = Dense(4,
                       activation=keras.layers.LeakyReLU(alpha=0.2),
                       name="FullConnectionStation_3")(station_y2)

    # 全部特征
    allin_y2 = Dense(4,
                     activation=keras.layers.LeakyReLU(alpha=0.2),
                     name="FullConnectionAIA_3")(allin_y2)
    # 模型层
    # 输入顺序： 气象 时滞 NDVI 时间 空间 AODs AOD
    # AOD + AODs
    model_aods = Model(
        inputs=[
            AODs_input,
            AOD_input],
        outputs=aods_y2)
    # AOD + 气象
    model_meteorology = Model(
        inputs=[
            Meteorology_input,
            AOD_input],
        outputs=meteorology_y2)
    # AOD + 时滞
    model_weather = Model(
        inputs=[Weather_input,
                AOD_input],
        outputs=weather_y2)
    # AOD + NDVI
    model_ndvi = Model(
        inputs=[
            Ndvi_input,
            AOD_input],
        outputs=ndvi_y2)
    # AOD + 时间
    model_time = Model(
        inputs=[
            Time_input,
            AOD_input],
        outputs=time_y2)
    # AOD + 空间
    model_station = Model(
        inputs=[
            Station_input,
            AOD_input],
        outputs=station_y2)
    # 全部特征
    model_allin = Model(
        inputs=[Meteorology_input,
                Weather_input,
                Ndvi_input,
                Time_input,
                Station_input,
                AODs_input,
                AOD_input],
        outputs=allin_y2)

    # 最后的融合
    # 捕捉的影响的融合层
    res_concat = concatenate([
        model_aods.output,
        model_meteorology.output,
        model_weather.output,
        model_ndvi.output,
        model_time.output,
        model_station.output,
        model_allin.output])

    # 全连接层 1
    res_x1 = Dense(8,activation=keras.layers.LeakyReLU(alpha=0.2),
                   name="ResFullConnectionResModelForLast")(res_concat)

    # 残差连接层
    res_residual_connection1 = Dense(8,activation=keras.layers.LeakyReLU(alpha=0.2),
                                     name="ResidualConnectionLast")(res_x1)

    res_residual_connection2 = Dense(
        8, activation=keras.layers.LeakyReLU(alpha=0.2),name="FullConnectionLast_RC")(res_residual_connection1)

    res_residual_output = add(
        [res_x1, res_residual_connection2], name="ResidualConnectionLast_Add")

    # 全连接层 2
    res_x2 = Dense(8,
                   activation=keras.layers.LeakyReLU(alpha=0.2),
                   name="FullConnectionLast_2")(res_residual_output)
    res_x3 = Dense(4,
                   activation=keras.layers.LeakyReLU(alpha=0.2),
                   name="FullConnectionLast_2x")(res_x2)
    # Dropout
    res_y = core.Dropout(rate=0.01, name="Res_Module")(res_x3)
    res_y2 = Dense(4,
                   activation=keras.layers.LeakyReLU(alpha=0.2),
                   name="FullConnectionLast_y")(res_y)

    # 最终融合结果
    res_outcome = Dense(
        1,
        activation=keras.layers.LeakyReLU(alpha=0.2),
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="sigmoid_FC")(res_y2)

    # 编译最终模型
    # 输入顺序： 气象 时滞 NDVI 时间 空间 AODs AOD
    model_last = Model(
        inputs=[
            Meteorology_input,
            Weather_input,
            Ndvi_input,
            Time_input,
            Station_input,
            AODs_input,
            AOD_input],
        outputs=res_outcome)

    model_last.compile(
        loss=['mean_absolute_error'],
        # optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.00001),
        # optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
        # optimizer=keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.00001),
        # optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
        optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
        # epsilon=None, decay=0.0, amsgrad=False),
        metrics=["accuracy"])

    # 格式转换
    data_sky_train = np.array(data_sky_train)
    data_t1_train = np.array(data_t1_train)
    data_ndvi_train = np.array(data_ndvi_train)
    data_time_train = np.array(data_time_train)
    data_station_train = np.array(data_station_train)
    data_aods_train = np.array(data_aods_train)
    data_aod_train = np.array(data_aod_train)

    data_sky_test = np.array(data_sky_test)
    data_t1_test = np.array(data_t1_test)
    data_ndvi_test = np.array(data_ndvi_test)
    data_time_test = np.array(data_time_test)
    data_station_test = np.array(data_station_test)
    data_aods_test = np.array(data_aods_test)
    data_aod_test = np.array(data_aod_test)

    # 运行
    # 输入顺序： 气象 时滞 NDVI 时间 空间 AODs AOD
    model_last.fit([
        data_sky_train,
        data_t1_train,
        data_ndvi_train,
        data_time_train,
        data_station_train,
        data_aods_train,
        data_aod_train
    ],
        data_pm_train,
        epochs=200,
        batch_size=5120)

    res = model_last.predict([data_sky_test,
                              data_t1_test,
                              data_ndvi_test,
                              data_time_test,
                              data_station_test,
                              data_aods_test,
                              data_aod_test])
    datares = res - data_pm_test
    datares.PM25 = datares.PM25.map(lambda x: abs(x))
    data_predt = pd.concat([datares, data_pm_test], axis=1)

    data_predt.columns = ["差值", '真']
    data_predt['差值'] = data_predt['差值'].map(lambda x: abs(x))
    data_predt['百分误'] = data_predt['差值'].div(data_predt["真"])
    data_predt['差值2'] = data_predt['差值'].map(lambda x: x ** 2)

    MAE = np.average(data_predt['差值'])
    RE = np.average(data_predt['百分误'])
    MSE = np.average(data_predt['差值2'])
    # os.system('shutdown -s -f -t 60')
    # print(np.average(data_predt['差值']))
    # print(np.average(data_predt['百分误']))
    # print(np.average(data_predt['差值2']))

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
a.to_excel('test100_sta.xlsx')
# os.system('shutdown -s -f -t 60')
