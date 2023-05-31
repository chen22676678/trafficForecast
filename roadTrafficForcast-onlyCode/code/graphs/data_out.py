"""
用于生成模型动态时间步的预测数据与实际数据，方便绘图
"""
import csv
from datetime import datetime

import flask
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 加载模型
print("Loading LSTM keras model...")
model_lstm = load_model('../../webapp/model/LSTM.h5')
print("LSTM model successfully loaded")
print("Loading GRU keras model...")
model_gru = load_model('../../webapp/model/GRU.h5')
print("GRU model successfully loaded")
while True:
    # 读入数据
    df = pd.read_csv("../../webapp/data/output_test_data_expand_Demo.csv", parse_dates=["datetime (Veh/5 Minutes)"],
                     index_col="datetime (Veh/5 Minutes)")
    # 分别对两个特征Flow、Avg kph使用不同的归一化器进行归一化
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))

    # 使用定义好的归一化器归一化数据
    df["Flow"] = flow_scaler.transform(df["Flow"].values.reshape(-1, 1))
    df["Avg kph"] = avg_kph_scaler.transform(df["Avg kph"].values.reshape(-1, 1))

    # 添加时间滞后值以构建模型输入特征
    lag = 48  # 1h->4;24h->96 控制时间步
    train, test = [], []
    for i in range(lag, len(df)):
        if i < 14109:  # 0.8->4703
            train.append(df[i - lag: i + 1])
        else:
            test.append(df[i - lag: i + 1])
    test = np.array(test)

    # 设置模型的输入
    x_test = test[:, :-1, [0, 1]]  # select Flow and Avg kph as input features
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))

    # 调用模型
    predicted_lstm = model_lstm.predict(x_test)
    predicted_gru = model_gru.predict(x_test)

    # 反向归一化模型预测数据
    predicted_lstm = flow_scaler.inverse_transform(predicted_lstm.reshape(-1, 1)).reshape(1, -1)[0]
    predicted_gru = flow_scaler.inverse_transform(predicted_gru.reshape(-1, 1)).reshape(1, -1)[0]

    # 从data/output_test_data_expand_Demo.csv获取最新的时间，即上一步保存的时间传给下一步
    df1 = pd.read_csv("../../webapp/data/output_test_data_expand_Demo.csv")
    col = df1.iloc[-1]['datetime (Veh/5 Minutes)']  # 获取最新一行的时间
    # print(col)
    # 读取对应日期的真实数据，由于前面读入df后进行了归一化，所以真实数据也需要反向归一化
    x = df.loc[col]
    print(x["Flow"])
    true = flow_scaler.inverse_transform(x["Flow"].reshape(-1, 1)).reshape(1, -1)[0]

    # 数据读入处理后索引变成了时间，所以需要计算时间步来提取模型的预测数据
    index_time = x.name
    pred_time = pd.Timestamp(index_time)

    # 2019-05-20 00:00 起始为0，计算预测时间点与起始时间点的时间步之差，间隔为15min
    time_point = datetime.strptime('2019-05-20 00:00', '%Y-%m-%d %H:%M')
    time_step = int((pred_time - time_point).total_seconds() / 60 / 5)  # 时间步之差，单位为5分钟
    pred_LSTM = predicted_lstm[time_step].astype(int)  # 4705-0
    pred_GRU = predicted_gru[time_step].astype(int)

    # 将预测数据写入文件以更新最后一个时间步
    # df_write = pd.read_csv("data/output_test_data_expand_Demo.csv")
    formatted_time = pred_time.strftime('%Y-%m-%d %H:%M')
    # 模型设置只预测输出了Flow，现在来生成avg kph
    min_avg_mph = 60
    max_avg_mph = 120
    # print(pred_LSTM)
    # df_write['Flow'].iloc[:-1] 去除最后一行，因为每次调用都会在文件最后更新一行仅包含时间的数据用于下一个时间步进行预测
    total_Flow = df1['Flow'].iloc[:-1].values
    avg_kph = (max_avg_mph - min_avg_mph) * \
              (1 - (pred_LSTM - total_Flow.min())  # pred_LSTM是本次时间节点的预测车流量
               / (total_Flow.max() - total_Flow.min())) + min_avg_mph
    # print(avg_kph)
    # # 根据所有数据获取正态分布的随机噪声，添加一些随机性
    random_noise = np.random.normal(0, 1, len(df1.iloc[:-1]))  # 以平均值为0，标准差为1的正态分布随机数为基础添加噪音
    avg_kph += random_noise[-1]
    # print(avg_kph)

    # # 创建新记录
    current_record = {"datetime (Veh/5 Minutes)": formatted_time,
                      "Flow": round(pred_LSTM),
                      "Avg kph": round(avg_kph)}
    #
    # # 将最后一行仅包含时间的数据更新
    df1.iloc[-1] = current_record
    df1[['Flow', 'Avg kph']] = df1[['Flow', 'Avg kph']].astype(int)
    # 将修改后的数据写回到文件中
    df1.to_csv("../../webapp/data/output_test_data_expand_Demo.csv", index=False)
    #
    # 同时加入下一个时间步的时间，用于调用模型
    # print(pred_time)
    delta = pd.Timedelta(minutes=5)

    next_time = (pred_time + delta).strftime('%Y-%m-%d %H:%M')
    # print(next_time)  # 时间步之差，单位为5分钟
    next_record = {"datetime (Veh/5 Minutes)": next_time}
    # # 添加一行下一个时间步的时间节点
    df1.loc[len(df1)] = next_record
    df1.to_csv("../../webapp/data/output_test_data_expand_Demo.csv", index=False)

    print("预测时间:" + str(pred_time))
    print("预测车流:  LSTM " + str(pred_LSTM) + '  GRU ' + str(pred_GRU))
    # 最后还需要将lstm与gru的预测和真实值存入新文件，方便绘制关于模型动态时间步的图像,避免影响预测，真实数据之后插入即可
    result = [{"datetime (Veh/5 Minutes)": formatted_time,
                      "LSTM": round(pred_LSTM),
                      "GRU": round(pred_GRU)}
    ]
    # 指定CSV文件的列名
    fields = ["datetime (Veh/5 Minutes)", "LSTM", "GRU"]
    # 指定CSV文件的路径和名称
    filename = "result.csv"

    # 使用with语句打开CSV文件，写入数据并关闭文件
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:  # 判断文件是否为空
            writer.writerow(fields)  # 写入表头
        for row in result:
            writer.writerow([row["datetime (Veh/5 Minutes)"],
                             round(row["LSTM"]),
                             round(row["GRU"])])
    # 判断是否需要继续循环
    # 如果条件满足，则继续循环；否则退出循环
    if col == '2019-05-21 00:00':  # 日期达到要求
        break
