# -*- coding: utf-8 -*-
# 日期: 2019/3/17 10:06
# 作者: xcl
# 工具：PyCharm

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit # 交叉验证所需的子集划分方法


model_file = "F:\\毕业论文程序\\缓存\\1-net.model"
data_file = "F:\\毕业论文程序\\整合数据\\各地区\\日均\\总地区.xlsx"
data = pd.read_excel(data_file)

# 自变量列
independent = ["AOD值", 'cloudCover', 'dewPoint', 'humidity', 'precipAccumulation', 'precipIntensity', 'pressure',
               'temperature', 'uvIndex', 'visibility', 'windSpeed']

# 因变量
dependent = ["日均PM2.5"]

# 打乱
data = shuffle(data)

# k折分组
kf = KFold(n_splits=10)  # 训练和测试 9:1

error = []
error_2 = []
for train, test in kf.split(data):
    # print("k折划分：%s %s" % (train.shape, test.shape))
    # print(data.iloc[test][dependent])  # loc根据文件索引, iloc根据第几行顺序, 从0开始
    x_train = data.iloc[train][independent].values
    x_test = data.iloc[test][independent].values
    y_train = data.iloc[train][dependent].values
    y_test = data.iloc[test][dependent].values
    # 隐藏层
    model = Sequential()
    model.add(Dense(input_dim=11, units=22))
    model.add(Activation("relu"))
    model.add(Dense(input_dim=22, units=1))

    model.compile(loss="mean_squared_error", optimizer="adam")  # 编译模型
    model.fit(x_train, y_train, epochs=10000, batch_size=100)  # 训练
    model.save_weights(model_file)

    # 预测
    data_pre = abs(model.predict(x_test) - y_test)
    data_pre2 = ((model.predict(x_test) - y_test)**2)
    error.append(data_pre.mean())
    error_2.append(data_pre2.mean())
print(np.average(error),  np.average(error_2), sep="\n")


# 误差     16.919624519175564
#          578.1610249489173