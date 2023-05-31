import numpy as np
import pandas as pd
import math

from keras import Input
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
import sklearn.metrics as metrics


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
    lag = 48  # 1h->4;24h->96 控制时间步
    # data = df.to_numpy()
    # print(int(len(data) * 0.8)) #4684
    train, test = [], []
    for i in range(lag, len(data)):
        if i < 14105:  # 0.8->4703
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


# 构建模型
def build_LSTM():
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=256, activation='relu', kernel_regularizer=l2(0.0001), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(LSTM(units=16))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss="mse", optimizer=optimizer, metrics=['mape'])
    return model


def build_GRU():
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(GRU(units=256, activation='relu', kernel_regularizer=l2(0.0001), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(GRU(units=16))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss="mse", optimizer=optimizer, metrics=['mape'])

    # model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    # # model.add(GRU(64, input_shape=(lag, 2), , activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(GRU(64, activation='relu', kernel_regularizer=l2(0.001), return_sequences=True))
    # model.add(Dropout(0.4))
    # model.add(Dense(1, activation='relu'))
    # optimizer = Adam(learning_rate=0.001)  # 增加在训练集上的表现能力
    # model.compile(loss="mse", optimizer=optimizer, metrics=['mape'])
    return model


# 评估模型
def evaluate_models(y_true, y_pred):
    y_true = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    # calculate the Mean Absolute Percentage Error
    sums = 0  # initialize value
    for i in range(len(y_pred)):
        tmp = abs(y_true[i] - y_pred[i]) / y_true[i]
        sums += tmp
    mape = sums * (100 / len(y_pred))
    # calculate variance score
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


if __name__ == '__main__':
    # 读入数据
    df = pd.read_csv("../Traffic_Data/output_test_data_expand.csv", parse_dates=["datetime (Veh/5 Minutes)"],
                     index_col="datetime (Veh/5 Minutes)")

    # 分别对两个特征Flow、Avg kph使用不同的归一化器进行归一化
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))

    # 使用定义好的归一化器归一化数据
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    # 添加时间滞后值以构建模型输入特征
    x_train, y_train, x_test, y_test = input_features(df)

    # 训练模型
    model_struct = "--"  # select LSTM or GRU
    if model_struct == "LSTM":
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))
        model = build_LSTM()
        monitor = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)
        hist = model.fit(x_train, y_train, batch_size=128, epochs=300, callbacks=[monitor],
                         validation_split=0.05)  # 128 0
        model.save('../../webapp/model/LSTM.h5')
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv('../../webapp/model/LSTM_loss.csv', encoding='utf-8', index=False)
    elif model_struct == "GRU":
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))
        model = build_GRU()
        monitor = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)
        hist = model.fit(x_train, y_train, batch_size=128, epochs=300, callbacks=[monitor], validation_split=0.05)
        model.save('../../webapp/model/GRU.h5')
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv('../../webapp/model/GRU_loss.csv', encoding='utf-8', index=False)

    # 装载模型以用于评估模型性能
    lstm = load_model('../../webapp/model/LSTM.h5')
    gru = load_model('../../webapp/model/GRU.h5')
    models = [lstm, gru]  #
    names = ['LSTM', 'GRU']  #
    y_test = denormalize_data(y_test, flow_scaler)
    y_preds = []
    for name, model in zip(names, models):
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
        predicted = model.predict(x_test)
        predicted = denormalize_data(predicted, flow_scaler)
        y_preds.append(predicted)
        print(name)
        evaluate_models(y_test, predicted)
