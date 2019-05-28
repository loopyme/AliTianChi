# python3
# -*- coding: utf-8 -*-
# @File    : choose_feature.py
# @Desc    :
# @Software: PyCharm
# @Time    : 19-5-22 下午11:12
# @Author  : Loopy
# @Doc     : http://api.loopy.tech/api/文档.html
# @Contact : 57658689098a@gmail.com

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBRegressor
import seaborn as sns
import warnings
import time

pd.set_option('max_colwidth', 200)
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)


def run_model(model, mesg, x, y):
    global log
    mse, i = [], 0
    for train, test in kfold.split(x):
        # split the dataset
        X_train, y_train = x.iloc[train], y.iloc[train]
        X_test, y_test = x.iloc[test], y.iloc[test]

        # fit & evaluate the model
        model.fit(X_train, y_train)
        mse.append(MSE(y_test, model.predict(X_test)))

    log.loc[mesg] = [np.cov(mse), np.mean(mse)]
    return {'cov': round(float(np.cov(mse)), 6), 'mean': round(float(np.mean(mse)), 6)}


def remove_feature(feature):
    global best_mean, threshold, X, Y

    basic_model = XGBRegressor()

    mean = run_model(basic_model, "remove {}".format(feature), X.drop([feature], axis=1), Y)['mean']
    mean_gain = best_mean - mean
    if mean_gain > threshold:
        print("接受 移除({}) mse增益:{} mse:{}".format(feature, round(mean_gain, 3), round(mean, 3)))
        best_mean = mean
        return True
    else:
        print("    拒绝 移除({}) mse增益:{}".format(feature, round(mean_gain, 3)))
        return False


def add_feature(f1, f2, op):
    global best_mean, threshold, X, Y

    basic_model = XGBRegressor()

    if op == '+':
        X[f1 + op + f2] = X[f1] + X[f2]
    elif op == '-':
        X[f1 + op + f2] = X[f1] - X[f2]
    elif op == '*':
        X[f1 + op + f2] = X[f1] * X[f2]
    elif op == '/':
        X[f1 + op + f2] = X[f1] / X[f2]

    mean = run_model(basic_model, "add {}".format(f1 + op + f2), X, Y)['mean']
    mean_gain = best_mean - mean
    if mean_gain > threshold:
        print(
            "{} 复验 新增({})  mse增益:{}  mse:{}".format(time.strftime("%m-%d %H:%M:%S", time.localtime()), f1 + op + f2,
                                                    round(mean_gain, 6), round(mean, 6)))
        recheck_gain = run_model(basic_model, "recheck_before", X.drop([f1 + op + f2], axis=1), Y)['mean'] - \
                       run_model(basic_model, "recheck {}".format(f1 + op + f2), X, Y)['mean']
        if recheck_gain > threshold:
            best_mean = mean
            print("{} 接受 新增({})  mse增益:{}  mse:{}".format(time.strftime("%m-%d %H:%M:%S", time.localtime()),
                                                          f1 + op + f2, round(recheck_gain, 6), round(mean, 6)))
            with open('./res.txt', 'a') as f:
                f.write(f1 + op + f2 + '\n')
            return True
        else:
            print(
                "{}      拒绝 新增({})  复验增益:{}".format(time.strftime("%m-%d %H:%M:%S", time.localtime()), f1 + op + f2,
                                                    round(recheck_gain, 6)))
            X = X.drop([f1 + op + f2], axis=1)
            return False
    else:
        print("{}      拒绝 新增({})  mse增益:{}  mse:{}".format(time.strftime("%m-%d %H:%M:%S", time.localtime()),
                                                           f1 + op + f2, round(mean_gain, 6), round(mean, 6)))
        X = X.drop([f1 + op + f2], axis=1)
        return False


if __name__ == '__main__':

    data = pd.read_csv('./data/zhengqi_train.txt', encoding='gbk', sep="\t")
    Y = data["target"]
    X = data.drop(['target'], axis=1)

    # init a kfold to split datasetk
    splits = 6
    kfold = KFold(n_splits=splits, shuffle=True, random_state=66)
    threshold = 0.002

    # model training and evaluating tool
    log = pd.DataFrame([], [], columns=['MSE_Cov', 'MSE_Mean'])

    features = list(X[1:1])
    basic_model = XGBRegressor()
    best_mean = run_model(basic_model, "初始化", X, Y)['mean']
    f_count = len(features)
    for i in range(1, f_count):
        f1 = features[i]
        for j in range(i + 1, f_count):
            f2 = features[j]
            for op in ['+', '-', '*', '/']:
                if add_feature(f1, f2, op):
                    if op == '+':
                        X[f1 + op + f2] = X[f1] + X[f2]
                    elif op == '-':
                        X[f1 + op + f2] = X[f1] - X[f2]
                    elif op == '*':
                        X[f1 + op + f2] = X[f1] * X[f2]
                    elif op == '/':
                        X[f1 + op + f2] = X[f1] / X[f2]

        log.to_csv('./{}.csv'.format(f1))
        print("{}已完成搜索".format(f1))

        log = pd.DataFrame([], [], columns=['MSE_Cov', 'MSE_Mean'])

        time.sleep(600)
