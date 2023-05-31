'''
数据量过少模型训练效果不好
使用线性插值扩充数据
'''
import pandas as pd

# 读入数据，使用datetime列作为索引
df = pd.read_csv('../Traffic_Data/output_test_data.csv', index_col='datetime (Veh/15 Minutes)', parse_dates=True)

# 重新采样为每5分钟一条记录
df = df.resample('5T').mean()

# 线性插值
df = df.interpolate(method='linear')
df[['Flow', 'Avg kph']] = df[['Flow', 'Avg kph']].astype(int)
# 修改索引列名称
df = df.rename_axis('datetime (Veh/5 Minutes)')
# 将索引转换为 DatetimeIndex 对象，并格式化为 "%Y-%m-%d %H:%M" 格式
df.index = pd.DatetimeIndex(df.index).strftime('%Y-%m-%d %H:%M')
# 输出结果
df.to_csv('../Traffic_Data/output_test_data_expand.csv')
df.to_csv('../../webapp/data/output_test_data_expand.csv')
# print(df)
