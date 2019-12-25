import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# 加载数据
data_set = pandas.read_csv("housing.csv")
# print("============================================================")
# print(data_set.head())
# print("============================================================")
# print(data_set.info())
# print("============================================================")
# print(data_set.describe())
# print("============================================================")
# data_set.hist(figsize=(15, 10))
# plt.show()

# 划分训练集与测试集
train_set, test_set = train_test_split(data_set, test_size=0.2, random_state=24)

# 可视化分析
# copy = train_set.copy()
# copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#                s=copy["population"] / 100, label="population", figsize=(10, 7),
#                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#                sharex=False)
# plt.show()
# corr_matrix = data_set.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 数据预处理


def initialize_data(data):
    data_i = data
    data_i = data_i.drop("ocean_proximity", axis=1)
    data_i = data_i.dropna(subset=["total_bedrooms"])
    return data_i


train_set = initialize_data(train_set)
test_set = initialize_data(test_set)
feature = train_set.drop("median_house_value", axis=1)
label = train_set["median_house_value"].copy()
ss = StandardScaler()
feature = ss.fit_transform(feature)

# 模型训练


def regression(name, model, fea, lab):
    model.fit(fea, lab)
    predictions = model.predict(fea)
    error = numpy.sqrt(mean_squared_error(lab, predictions))
    score = cross_val_score(model, fea, lab, cv=10)
    print("["+name+"]")
    print("mean_squared_error = ", error)
    print("mean_score = ", score.mean())


# 1 线性回归
# regression("LinearRegression", LinearRegression(), feature, label)
# 2 决策树回归
# regression("DecisionTreeRegression", DecisionTreeRegressor(), feature, label)
# 3 SVR
# regression("SVR", SVR(), feature, label)
# 4 KNN回归
# regression("KNeighborsRegression", KNeighborsRegressor(), feature, label)
# 5 Bagging回归
# regression("BaggingRegression", BaggingRegressor(), feature, label)
# 6 Random Forest回归
# regression("RandomForestRegression", RandomForestRegressor(), feature, label)
# 7 Adaboost回归
# regression("AdaBoostRegression", AdaBoostRegressor(), feature, label)
# 8 GBDT回归
# regression("GradientBoostingRegression", GradientBoostingRegressor(), feature, label)

# 模型调优
parameters_grid = [
    {"n_estimators": [400, 450, 500], "max_features": [6, 8]}
]


def improvement(name, model, fea, lab):
    grid_search = GridSearchCV(model, parameters_grid, cv=5)
    grid_search.fit(fea, lab)
    print("[" + name + "]")
    print("best_params:", grid_search.best_params_)
    print("best_estimator:", grid_search.best_estimator_)
    print("best_score:", grid_search.best_score_)


# Bagging调优
# improvement("BaggingRegression", BaggingRegressor(), feature, label)
# Random Forest调优
# improvement("RandomForestRegression", RandomForestRegressor(), feature, label)
# AdaBoost调优
# improvement("AdaBoostRegression", AdaBoostRegressor(), feature, label)
# GBDT调优
# improvement("GradientBoostingRegression", GradientBoostingRegressor(), feature, label)

# 模型评估


def final_predict(name, model, fea1, lab1, fea2, lab2):
    model.fit(fea1, lab1)
    final_prediction = model.predict(fea2)
    final_error = numpy.sqrt(mean_squared_error(lab2, final_prediction))
    print("[" + name + "]")
    print("mean_squared_error = ", final_error)


feature_test = test_set.drop("median_house_value", axis=1)
label_test = test_set["median_house_value"].copy()
feature_test = ss.transform(feature_test)
# 1 线性回归
final_predict("LinearRegression", LinearRegression(), feature, label, feature_test, label_test)
# 2 决策树回归
final_predict("DecisionTreeRegression", DecisionTreeRegressor(), feature, label, feature_test, label_test)
# 3 SVR
final_predict("SVR", SVR(), feature, label, feature_test, label_test)
# 4 KNN回归
final_predict("KNeighborsRegression", KNeighborsRegressor(), feature, label, feature_test, label_test)
# 5 Bagging回归
final_predict("BaggingRegression", BaggingRegressor(n_estimators=40,max_features=6), feature, label,
              feature_test, label_test)
# 6 Random Forest回归
final_predict("RandomForestRegression", RandomForestRegressor(n_estimators=100,max_features=6), feature, label,
              feature_test, label_test)
# 7 Adaboost回归
final_predict("AdaBoostRegression", AdaBoostRegressor(n_estimators=100), feature, label,
              feature_test, label_test)
# 8 GBDT回归
final_predict("GradientBoostingRegression", GradientBoostingRegressor(n_estimators=500,max_features=6), feature, label,
              feature_test, label_test)
