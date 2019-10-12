# -*- coding: utf-8 -*-
# 日期: 2019/6/9 15:49
# 作者: xcl
# 工具：PyCharm


from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
from sklearn.utils import check_random_state
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from sklearn.model_selection import KFold
import datetime  # 程序耗时
import os

print('#######读取数据#######')
start_time = datetime.datetime.now()
data = pd.read_excel("相邻位置仅留PM和T-1.xlsx")

# 删除字符串列
c = "str"
for columns_type in data.columns:
    if data.iloc[1][columns_type].__class__ == c.__class__:
        del data[columns_type]

print('#######设置变量#######')

# 自变量列
independent_raw = list(data.columns)
independent = list(set(independent_raw) - set(["日均PM2.5", "日期", "监测站",]))
# 因变量
dependent = ["日均PM2.5"]

print('#######删除“0”列#######')
for c_0 in independent:
    if np.var(data[c_0]) == 0:
        del data[c_0]
# 更新自变量列
independent_raw = list(data.columns)
independent = list(set(independent_raw) - set(["日均PM2.5", "日期", "监测站"]))
independent = ["AOD值", 'cloudCover', 'dewPoint', 'humidity', 'precipAccumulation', 'precipIntensity', 'pressure',
               'temperature', 'uvIndex', 'visibility', 'windSpeed', 'windBearing']
T_1 = ["AOD值-t-1", 'cloudCover-t-1', 'dewPoint-t-1', 'humidity-t-1', 'precipAccumulation-t-1', 'precipIntensity-t-1', 'pressure-t-1',
               'temperature-t-1', 'uvIndex-t-1', 'visibility-t-1', 'windSpeed-t-1', 'windBearing-t-1', "日均PM2.5-t-1"]
PM_list = ["A1-日均PM2.5-MEAN-t-1", "A2-日均PM2.5-MEAN-t-1", "A3-日均PM2.5-MEAN-t-1", "A4-日均PM2.5-MEAN-t-1",
           "A5-日均PM2.5-MEAN-t-1", "A6-日均PM2.5-MEAN-t-1", "A7-日均PM2.5-MEAN-t-1", "A8-日均PM2.5-MEAN-t-1",
           "B1-日均PM2.5-MEAN-t-1", "B2-日均PM2.5-MEAN-t-1", "B3-日均PM2.5-MEAN-t-1", "B4-日均PM2.5-MEAN-t-1",
           "B5-日均PM2.5-MEAN-t-1", "B6-日均PM2.5-MEAN-t-1", "B7-日均PM2.5-MEAN-t-1", "B8-日均PM2.5-MEAN-t-1"]
independent = list(set(independent) | set(T_1))  # 合集
independent = list(set(independent) | set(PM_list))  # 合集
###### 选择 AOD ！= 0 的行
data = data[data["AOD值"] > 0]
'''
for llll in data.columns:
    if "AOD" in llll:
        data = data[data[llll] > 0]
print(data)
'''
print('#######打乱数据#######')
# 打乱
data = shuffle(data)
print(len(data))

print('#######进行K折分组#######')
# k折分组,训练和测试 9:1
kf = KFold(n_splits=10)  # 参数shuffle=True

error_AME = []
error_MSE = []

print('#######开始计算#######')
for train, test in kf.split(data):
    # 参数设置
    # 一个隐藏层 22个隐藏单元
    mlp = MLPRegressor(hidden_layer_sizes=(18, 4,), solver='adam', max_iter=10000, learning_rate="adaptive",
                       activation="relu", learning_rate_init=0.01)
    rng = check_random_state(0)
    # 划分
    x_train = data.iloc[train][independent].values
    x_test = data.iloc[test][independent].values
    y_train = data.iloc[train][dependent].values.ravel()
    y_test = data.iloc[test][dependent].values.ravel()
    ensemble = AdaBoostRegressor(base_estimator=mlp, learning_rate=0.001,
                                 loss='linear').fit(x_train, y_train)  # 先别改参数
    res = ensemble.predict(x_test)
    # print(res, y_test)
    # 格式转换
    res = pd.DataFrame(res)
    y_test = pd.DataFrame(y_test)
    # 相同索引方便合并
    res.index = y_test.index
    data_pred = pd.concat([res, y_test], axis=1)
    data_pred.columns = ["pre", "true"]
    # 计算误差
    e_AME = abs(data_pred["pre"] - data_pred["true"]).mean()
    # print("AME误差:", e)
    e_MSE = ((data_pred["pre"] - data_pred["true"]) ** 2).mean()
    error_AME.append(e_AME)
    error_MSE.append(e_MSE)
print("交叉验证后的平均AME误差值:", np.average(error_AME), "\n", "预测结果的标准差", np.std(error_AME))
print("交叉验证后的平均MSE误差值:", np.average(error_MSE), "\n", "预测结果的标准差", np.std(error_MSE))
end_time = datetime.datetime.now()
print("总耗时:"+str(end_time - start_time))

a = np.average(error_AME)
b = np.std(error_AME)
c = np.average(error_MSE)
d = np.std(error_MSE)

e = list([a,b,c,d])
e = pd.Series(e)
e.to_excel("outcome_raw.xlsx")

#os.system('shutdown -s -f -t 60')