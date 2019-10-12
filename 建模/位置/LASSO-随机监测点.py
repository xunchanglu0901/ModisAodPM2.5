# -*- coding: utf-8 -*-
# 日期: 2019/3/18 9:26
# 作者: xcl
# 工具：PyCharm

"""
由于过多的虚拟变量导致结果爆炸。
这里引入了【id2】按四个省份设置了四个虚拟变量。
误差稳定在19+ 20-
"""
import random,copy
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle

import numpy as np
from sklearn.utils import check_random_state
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from sklearn.linear_model import Lasso


# 读取
input_path = 'D:\\雨雪+2018_new_pm_aod_interpolate线性2.xlsx'
data_all = pd.read_excel(input_path, index_col='日期')
# 去空
data_all = data_all.dropna()
data_ts_df = data_all[['tm_mon', 'tm_mday',
                       'tm_wday', 'tm_yday', 'tm_week', 'id','id2']]
# 虚拟变量
for ccc in data_ts_df.columns:
    data_ts_df[ccc] = data_ts_df[ccc].map(lambda x: str(x))
data_get_dummies1 = pd.get_dummies(data_ts_df[['tm_mon']], drop_first=True)
data_get_dummies3 = pd.get_dummies(data_ts_df[['id']], drop_first=True)
data_dummies = pd.concat([data_get_dummies1,
                          data_get_dummies3,
                          data_ts_df[['tm_mon']],
                          data_ts_df[['id']],
                          data_ts_df[['id2']]],
                         axis=1)

# 去掉不标准化列
data_to_std = data_all.drop(
    ['tm_mon', 'tm_mday', 'tm_wday', 'tm_yday', 'tm_week','id','id2' ], axis=1)


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
# 误差

MAE_list = []
RE_list = []
MSE_list = []
for t_numb in range(0, 50):
    # 划分
    idlist = list(range(1, 153))
    slice1 = random.sample(idlist, 38)  # 从list中随机获取5个元素，作为一个片断返回
    slice2 = []
    for idx in idlist:
        if idx not in slice1:
            idx = str(idx)
            slice2.append(idx)
    slice1 = [str(j) for j in slice1]
    # 划分不标准化下的训练集测试集, 用于检验
    data_test2 = data_out2[data_out2["id"].isin(slice1)]
    # print(data_test2.PM25)  # 这才是真实值

    # 划分标准化后的训练集测试集, 用于训练
    data_test = data_out[data_out["id"].isin(slice1)]
    data_train = data_out[data_out["id"].isin(slice2)]
    # print(data_train.index)
    # 自变量
    '''
                       'tm_mon_10',
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
    '''
    independent = [
                   'AOD_0',
        'tm_mon_10',
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
                   'AOD_0_T1',
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
                   'NDVI_0',
                   'cloudCover',
                   'dewPoint',
                   'humidity',
                   'sunTime',
                   'visibility',
                   'windBearing',
                   'windGust',
                   'windSpeed',
                   'apparentTemperature',
                   'temperature',
                   'tempMM',
                   'tempHL',
                   'atempMM',
                   'atempHL',
                   'pressure',
                   'precipIntensity',
                   'precipAccumulation',
                   'AOD_1',
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
                   'AOD_16', ]
    for clo in data_get_dummies3.columns:
        independent.append(clo)
    # 因变量
    dependent = ["PM25"]

    # 打乱
    data = shuffle(data_out)

    # 参数设置
    alpha = 0.1
    lasso = Lasso(alpha=alpha)
    # 划分

    x_train = data_train[independent].values
    x_test = data_test[independent].values
    y_train = data_train[dependent].values.ravel()
    y_test = data_test[dependent].values.ravel()
    res = lasso.fit(x_train, y_train).predict(x_test)


    datares = res - y_test
    datares = pd.DataFrame(datares,index=data_test.index, columns = ['PM25'])
    datares.PM25 = datares.PM25.map(lambda x: abs(x))
    data_predt = pd.concat([datares, pd.DataFrame(y_test,index=data_test.index, columns = ['PM25'])], axis=1)
    data_predt.columns = ["差值", '真']
    data_predt['差值'] = data_predt['差值'].map(lambda x: abs(x))
    data_predt['百分误'] = data_predt['差值'].div(data_predt["真"])
    data_predt['差值2'] = data_predt['差值'].map(lambda x: x ** 2)
    e_AME = np.average(data_predt['差值'])
    e_RE = np.average(data_predt['百分误'])
    e_MSE = np.average(data_predt['差值2'])
    # 还原，反标准化
    res2 = [float((j * std_pm) + mean_pm) for j in res]
    res2 = pd.DataFrame(res2, index=data_test.index, columns=['PM25'])

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
a.to_excel('LASSO_随机监测点_标准化.xlsx')
# os.system('shutdown -s -f -t 60')
