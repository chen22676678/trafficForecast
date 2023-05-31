"""
该程序每调用一次会更新一次数据，用于演示模型在未来时间点的预测效果
帮助判断模型在预测多少个时间步，前面48个时间步的累计误差达到了触发模型迭代的阈值
所以此时演示界面的 True number 会变成 NaN
不建议直接使用清洗好的数据
4个小时即48个时间步后预测数据完全基于模型预测的数据进行预测
"""

from datetime import datetime

import flask
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    # 读入数据
    df = pd.read_csv("data/output_test_data_expand_Demo.csv", parse_dates=["datetime (Veh/5 Minutes)"],
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

    # 加载模型
    print("Loading LSTM keras model...")
    model_lstm = load_model('../webapp/model/LSTM.h5')
    print("LSTM model successfully loaded")
    print("Loading GRU keras model...")
    model_gru = load_model('../webapp/model/GRU.h5')
    print("GRU model successfully loaded")

    # 调用模型
    predicted_lstm = model_lstm.predict(x_test)
    predicted_gru = model_gru.predict(x_test)

    # 反向归一化模型预测数据
    predicted_lstm = flow_scaler.inverse_transform(predicted_lstm.reshape(-1, 1)).reshape(1, -1)[0]
    predicted_gru = flow_scaler.inverse_transform(predicted_gru.reshape(-1, 1)).reshape(1, -1)[0]

    # 前端请求
    if flask.request.method == 'GET':
        return flask.render_template('index.html')
    if flask.request.method == 'POST':
        # 接收文本框输入
        date = flask.request.form['date']
        time = flask.request.form['time']
        col = date + " " + time

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
        df_write = pd.read_csv("data/output_test_data_expand_Demo.csv")
        formatted_time = pred_time.strftime('%Y-%m-%d %H:%M')
        # 模型设置只预测输出了Flow，现在来生成avg kph
        min_avg_mph = 60
        max_avg_mph = 120
        # print(pred_LSTM)
        # df_write['Flow'].iloc[:-1] 去除最后一行，因为每次调用都会在文件最后更新一行仅包含时间的数据用于下一个时间步进行预测
        total_Flow = df_write['Flow'].iloc[:-1].values
        avg_kph = (max_avg_mph - min_avg_mph) * \
                  (1 - (pred_LSTM - total_Flow.min())  # pred_LSTM是本次时间节点的预测车流量
                   / (total_Flow.max() - total_Flow.min())) + min_avg_mph
        # print(avg_kph)
        # # 根据所有数据获取正态分布的随机噪声，添加一些随机性
        random_noise = np.random.normal(0, 1, len(df_write.iloc[:-1]))  # 以平均值为0，标准差为1的正态分布随机数为基础添加噪音
        avg_kph += random_noise[-1]
        # print(avg_kph)

        # # 创建新记录
        current_record = {"datetime (Veh/5 Minutes)": formatted_time,
                          "Flow": round(pred_LSTM),
                          "Avg kph": round(avg_kph)}
        #
        # # 将最后一行仅包含时间的数据更新
        df_write.iloc[-1] = current_record
        df_write[['Flow', 'Avg kph']] = df_write[['Flow', 'Avg kph']].astype(int)
        # 将修改后的数据写回到文件中
        df_write.to_csv("../webapp/data/output_test_data_expand_Demo.csv", index=False)
        #
        # 同时加入下一个时间步的时间，用于调用模型
        # print(pred_time)
        delta = pd.Timedelta(minutes=5)

        next_time = (pred_time + delta).strftime('%Y-%m-%d %H:%M')
        # print(next_time)  # 时间步之差，单位为5分钟
        next_record = {"datetime (Veh/5 Minutes)": next_time}
        # # 添加一行下一个时间步的时间节点
        df_write.loc[len(df_write)] = next_record
        df_write.to_csv("../webapp/data/output_test_data_expand_Demo.csv", index=False)

        percentege_lstm = int((pred_LSTM / 1000) * 100)
        print("预测时间:" + str(pred_time))
        print("预测车流:" + str(pred_LSTM))
        return flask.render_template('index.html', true_val=true, pred_val_LSTM=pred_LSTM,
                                     date=date, time=time,
                                     pred_val_GRU=pred_GRU, percentage=percentege_lstm)


if __name__ == '__main__':
    app.run()
