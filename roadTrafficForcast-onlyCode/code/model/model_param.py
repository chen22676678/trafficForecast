import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
'''
代码迭代后，该文件代码就没有修改了，不适用于最佳参数获取
'''

# 定义LSTM模型
def create_model(learning_rate=0.001, units=64, dropout=0.2, l2_reg=0.001):
    model = Sequential()
    model.add(LSTM(units, input_shape=(12, 2), return_sequences=True, kernel_regularizer=l2(l2_reg)))
    model.add(LSTM(units, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='relu'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[mean_absolute_percentage_error])
    return model

# 加载数据
df = pd.read_csv("webapp/data/output_test_data.csv", parse_dates=["datetime (Veh/15 Minutes)"], index_col="datetime (Veh/15 Minutes)")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1)).fit(df[["Flow", "Avg kph"]])
df[["Flow", "Avg kph"]] = scaler.transform(df[["Flow", "Avg kph"]])

# Practicing with different time lag (look back) values to optimize the models
lag = 12
data = df.to_numpy()
train, test = [], []
for i in range(lag, len(data)):
    if i < int(len(data) * 0.8):  # 0.8
        train.append(data[i - lag: i + 1])
    else:
        test.append(data[i - lag: i + 1])

train = np.array(train)
test = np.array(test)

# Shuffle data (stateless case)
np.random.shuffle(train)
X_train = train[:, :-1, [0, 1]]  # select Flow and Avg kph as input features
y_train = train[:, -1, 0]  # select Flow as output
X_test = test[:, :-1, [0, 1]]
y_test = test[:, -1, 0]
# X_train, y_train, X_test, y_test = load_data()

# 定义参数范围
param_grid = {
    'learning_rate': [0.001, 0.0001],
    'units': [32, 64],
    'dropout': [0.2, 0.3],
    'l2_reg': [0.001, 0.01]
}

# 定义交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 定义KerasRegressor，用于调用Keras模型
kr = KerasRegressor(build_fn=create_model, epochs=10, batch_size=64, verbose=0)

# 定义GridSearchCV，用于调用交叉验证
grid = GridSearchCV(estimator=kr, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error')

# 执行交叉验证
grid_result = grid.fit(X_train, y_train)

# 输出最佳参数
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# 输出所有参数的评分
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
