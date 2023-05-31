"""
不会对数据进行更新，仅用于历史数据预测对比
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
    df = pd.read_csv("data/output_test_data_expand.csv", parse_dates=["datetime (Veh/5 Minutes)"],
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
        print(x)
        print(x["Flow"])
        true = int(flow_scaler.inverse_transform(x["Flow"].reshape(-1, 1)).reshape(1, -1)[0])

        # 数据读入处理后索引变成了时间，所以需要计算时间步来提取模型的预测数据
        index_time = x.name
        pred_time = pd.Timestamp(index_time)

        # 2019-05-20 00:00 起始为0，计算预测时间点与起始时间点的时间步之差，间隔为15min
        time_point = datetime.strptime('2019-05-20 00:00', '%Y-%m-%d %H:%M')
        time_step = int((pred_time - time_point).total_seconds() / 60 / 5)  # 时间步之差，单位为5分钟
        pred_LSTM = predicted_lstm[time_step].astype(int)  # 4705-0
        pred_GRU = predicted_gru[time_step].astype(int)
        percentege = int((pred_LSTM / 1000) * 100)

        print("预测时间:" + str(pred_time))
        print("预测车流:" + str(pred_LSTM))
        return flask.render_template('index.html', true_val=true, pred_val_LSTM=pred_LSTM,
                                     date=date, time=time,
                                     pred_val_GRU=pred_GRU, percentage=percentege)


if __name__ == '__main__':
    app.run()
