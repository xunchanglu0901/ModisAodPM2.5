
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.ensemble import BaggingRegressor
import pandas as pd


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

x = data[independent].values
y = data[dependent].values


mlp = MLPClassifier(hidden_layer_sizes=(21,), max_iter=500)  # 一个隐藏层 22个隐藏单元

rng = check_random_state(0)
x_train, x_test, y_train, y_test = train_test_split(data[independent],
                                                    data[dependent],
                                                    random_state=rng)
ensemble = BaggingRegressor(base_estimator=mlp,
                            max_features=1.0,
                            bootstrap_features=False,
                            random_state=rng).fit(x_train, y_train)
res = ensemble.predict(x_test)
print(res, y_test)
res = pd.DataFrame(res)
res.index = y_test.index

data_pred = pd.concat([res, y_test], axis=1)

data_pred.columns = ["pre", "true"]

print(abs(data_pred["pre"]-data_pred["true"]).mean())