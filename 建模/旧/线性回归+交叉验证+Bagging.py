# -*- coding: utf-8 -*-
# 日期: 2019/3/18 9:34
# 作者: xcl
# 工具：PyCharm



from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.utils import check_random_state
from sklearn.ensemble import BaggingRegressor
import pandas as pd
from sklearn.model_selection import KFold

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


error_AME = []
error_MSE = []
for train, test in kf.split(data):
    # 参数设置
    # 一个隐藏层 22个隐藏单元
    mlp = LinearRegression(fit_intercept=True)
    rng = check_random_state(0)
    # 划分
    x_train = data.iloc[train][independent].values
    x_test = data.iloc[test][independent].values
    y_train = data.iloc[train][dependent].values.ravel()
    y_test = data.iloc[test][dependent].values.ravel()
    ensemble = BaggingRegressor(base_estimator=mlp,
                                max_features=1.0,
                                bootstrap_features=False,
                                random_state=rng).fit(x_train, y_train)
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

