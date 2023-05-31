import datetime

import numpy as np
import pandas as pd
import math

from matplotlib.dates import HourLocator, MinuteLocator
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
'''
此代码绘制在静态时间步、动态时间步的数据预测效果
'''
# 数据归一化
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

# 反向归一化
def denormalize_data(data, scaler):
    data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(1, -1)[0]
    return data

# 添加时间滞后值以构建模型输入特征
def input_features(data):
    lag = 48 #1h->4;24h->96 控制时间步
    train, test = [], []
    # print(len(data)*0.8)
    for i in range(lag, len(data)):
        if i < 14109:  # 0.8->14105-> 5.19 23:30
            train.append(data[i - lag: i + 1])
        else:
            test.append(data[i - lag: i + 1])

    train = np.array(train)
    test = np.array(test)

    # 设置模型的输入输出
    x_train = train[:, :-1, [0, 1]]  # select Flow and Avg kph as input features
    y_train = train[:, -1, [0]]  # select Flow as output
    x_test = test[:, :-1, [0, 1]]  # select Flow and Avg kph as input features
    y_test = test[:, -1, [0]]  # select Flow as output
    return x_train, y_train, x_test, y_test

#绘图
# 静态时间步
def plot_LSTM_oneDay(y_true, y_pred):
    # 定义日期字符串 d，表示起始时间点
    d = '2019-05-20 00:00'
    # 使用 Pandas 库生成一个包含 864（3天） 个时间点的时间序列，频率为每 15 分钟一个时间点
    x = pd.date_range(d, periods=864, freq='5min')
    # 创建一个大小为 20x10 的画布
    fig = plt.figure(figsize=(20, 10))
    # 在画布中添加一个子图
    ax = fig.add_subplot(111)
    # 在子图中绘制真实值的折线，标签为 'True Data'
    ax.plot(x, y_true, label='True')
    # 对于模型预测的每个结果，绘制其预测值的折线，标签为模型的名称
    for name, y_pred in zip(['LSTM'], y_preds):
        ax.plot(x, y_pred[: 864], label=name)
    # 在图形中添加图例
    plt.legend(fontsize=17)
    plt.grid(True)

    # 设置横轴刻度文字大小为17
    plt.xticks(fontsize=17)
    # 设置纵轴刻度文字大小为17
    plt.yticks(fontsize=17)
    # 设置标签文字大小为18
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Flow', fontsize=18)
    # 将横轴的时间数据格式化为小时:分钟的形式
    date_format = mpl.dates.DateFormatter("%H:%M")
    # 自动调整横轴标签的显示方式，避免标签重叠
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.xlim(pd.Timestamp(d), pd.Timestamp('2019-05-23 00:00'))
    plt.ylim(0)
    fig.savefig('LSTM_static_3Day.png')
    plt.show()

def plot_GRU_oneDay(y_true, y_pred):
    # 定义日期字符串 d，表示起始时间点
    d = '2019-05-20 00:00'
    # 使用 Pandas 库生成一个包含 864（3天） 个时间点的时间序列，频率为每 15 分钟一个时间点
    x = pd.date_range(d, periods=864, freq='5min')
    # 创建一个大小为 20x10 的画布
    fig = plt.figure(figsize=(20, 10))
    # 在画布中添加一个子图
    ax = fig.add_subplot(111)
    # 在子图中绘制真实值的折线，标签为 'True Data'
    ax.plot(x, y_true, label='True')
    # 对于模型预测的每个结果，绘制其预测值的折线，标签为模型的名称
    for name, y_pred in zip(['GRU'], y_preds):
        ax.plot(x, y_pred[: 864], label=name)
    # 在图形中添加图例
    plt.legend(fontsize=17)
    plt.grid(True)
    # 设置横轴刻度文字大小为17
    plt.xticks(fontsize=17)
    # 设置纵轴刻度文字大小为17
    plt.yticks(fontsize=17)
    # 设置标签文字大小为18
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Flow', fontsize=18)
    # 将横轴的时间数据格式化为小时:分钟的形式
    date_format = mpl.dates.DateFormatter("%H:%M")
    # 自动调整横轴标签的显示方式，避免标签重叠
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.xlim(pd.Timestamp(d), pd.Timestamp('2019-05-23 00:00'))
    plt.ylim(0)
    fig.savefig('GRU_static_3Day.png')
    plt.show()

# 动态时间步
def plot_LSTM_GRU_dynamic_oneDay(df1):
    # df1["datetime (Veh/5 Minutes)", "True", "LSTM", "GRU"]
    # 绘制折线图
    d = '2019-05-20 00:00'
    # 使用 Pandas 库生成一个包含 145 个时间点的时间序列，频率为每 5 分钟一个时间点
    x = pd.date_range(d, periods=145, freq='5min')
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, df1["True"], label='True')
    ax.plot(x, df1["LSTM"], label='LSTM')
    ax.plot(x, df1["GRU"], label='GRU')
    plt.legend(fontsize=17)
    plt.grid(True)

    # 设置横轴刻度文字大小为17
    plt.xticks(fontsize=17)
    # 设置纵轴刻度文字大小为17
    plt.yticks(fontsize=17)
    # 设置标签文字大小为18
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Flow', fontsize=18)

    # 将横轴的时间数据格式化为小时:分钟的形式
    date_format = mpl.dates.DateFormatter("%H:%M")
    # 自动调整横轴标签的显示方式，避免标签重叠
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    # 设置x轴范围
    plt.xlim(pd.Timestamp('2019-05-20 00:00'), pd.Timestamp('2019-05-20 12:00'))
    plt.ylim(0)
    fig.savefig('LSTM_GRU_dynamic_12H.png')
    plt.show()

if __name__ == '__main__':
    # 读入数据
    df = pd.read_csv("../Traffic_Data/output_test_data_expand.csv", parse_dates=["datetime (Veh/5 Minutes)"],
                     index_col="datetime (Veh/5 Minutes)")

    # 分别对两个特征Flow、Avg kph使用不同的归一化器进行归一化
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))

    #使用定义好的归一化器归一化数据
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    # 添加时间滞后值以构建模型输入特征
    x_train, y_train, x_test, y_test = input_features(df)

    # 装载模型以用于评估模型性能和绘图
    # 分开绘图
    lstm = load_model('../../webapp/model/LSTM.h5')
    gru = load_model('../../webapp/model/GRU.h5')
    models = [lstm]  #lstm,gru
    names = ['LSTM']  #'LSTM','GRU'
    y_test = denormalize_data(y_test, flow_scaler)
    y_preds = []
    for name, model in zip(names, models):
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
        predicted = model.predict(x_test)
        predicted = denormalize_data(predicted, flow_scaler)
        y_preds.append(predicted)
        print(name)
    # print(y_test[: 288]) # 定位app.py预测数据点位14109
    # 分别调用模型去绘制两幅图结合两种模型绘制静态数据三天

    plot_LSTM_oneDay(y_test[: 864], y_preds[: 864])  # 三天的静态时间步预测，
    # plot_GRU_oneDay(y_test[: 864], y_preds[: 864])  # 三天的静态时间步预测

    # 绘图应该使用动态时间步生成的数据与原始存在的数据做对比
    #动态数据一幅分图绘制一天
    # df_dynamic = pd.read_csv('merged_file.csv')
    # plot_LSTM_GRU_dynamic_oneDay(df_dynamic)
    # plot_GRU_dynamic_oneDay(df_dynamic)