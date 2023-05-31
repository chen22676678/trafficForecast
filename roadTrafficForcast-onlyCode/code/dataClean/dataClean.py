'''
数据清洗

完成
'''
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
# 读取数据文件
df = pd.read_csv('../Traffic_Data/DailyStandard_Report_1_8978_01_04_2019_31_05_2019.csv')

# 提取所需的列
df = df[['Report Date', 'Time Period Ending', 'Avg mph', 'Total Volume']]

# 将 Report Date 和 Time Period Ending 合并为一个日期时间列
df['datetime (Veh/15 Minutes)'] = df['Report Date'].str.slice(0, 10) + ' ' + df['Time Period Ending'].str.slice(0, 5)
df['datetime (Veh/15 Minutes)'] = pd.to_datetime(df['datetime (Veh/15 Minutes)'], format='%d/%m/%Y %H:%M')

# 删除原来的 Report Date 和 Time Period Ending 列
df = df.drop(['Report Date', 'Time Period Ending'], axis=1)
df = df.groupby('datetime (Veh/15 Minutes)').mean()

# 格式化输出结果
df['Flow'] = df['Total Volume'].astype(int)
# df['Avg kph'] = (df['Avg mph'] * 1.60934).astype(int)   # 将 Avg mph 转换为公里/小时

# 计算Avg mph
min_avg_mph = 60
max_avg_mph = 120

df['Avg kph'] = (max_avg_mph - min_avg_mph) * (1 - (df['Flow'] - df['Flow'].min()) / (df['Flow'].max() - df['Flow'].min())) + min_avg_mph
# print(df['Avg kph'])
# 添加一些随机性
random_noise = np.random.normal(0, 1, len(df)) # 以平均值为0，标准差为1的正态分布随机数为基础添加噪音
df['Avg kph'] += random_noise

# 将Avg mph列的值限制在60-120之间
df['Avg kph'] = np.clip(df['Avg kph'], min_avg_mph, max_avg_mph).astype(int)

df = df.drop(['Total Volume', 'Avg mph'], axis=1)
df = df.reset_index()
df['datetime (Veh/15 Minutes)'] = df['datetime (Veh/15 Minutes)'].dt.strftime('%Y-%m-%d %H:%M')
#将时间调整为整点时间，方便后期绘图展示
for i, time_str in enumerate(df['datetime (Veh/15 Minutes)']): # enumerate 方便获取行索引
    time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
    new_time_obj = time_obj + timedelta(minutes=1)
    new_time_str = new_time_obj.strftime('%Y-%m-%d %H:%M')
    # 如果时间增加之后跨越了一天，则对日期也进行加1天操作
    # if new_time_obj.day != time_obj.day:
    #     new_time_obj += timedelta(days=1)
    # new_time_str = new_time_obj.strftime('%Y-%m-%d %H:%M')
    # 修改df
    df.loc[i, 'datetime (Veh/15 Minutes)'] = new_time_str

# 通畅率等于正常行驶情况下车辆数/容纳量
# 这个容纳数我们无从得知，因此可以采取一个取高峰期时间段的车流量平均值来作为一个容纳量的估计     290
'''
nlargest 函数中的第一个参数指定要获取前几条最大值
第二个参数指定要获取最大值的列名。然后，['xxx'] 表示获取 'xxx' 列的数据。最后，使用 head 函数获取前二十条数据。
'''
print('---------------------------------------------\n车流量较大值的前十个值：')
print(df.nlargest(10, 'Flow')[['datetime (Veh/15 Minutes)', 'Avg kph', 'Flow']].to_string(index=False))
#
# print('---------------------------------------------\n平均车速的前五个众数：')
# # 计算 Avg kph 列的最大的众数
# counts = df['Avg kph'].value_counts()
# # 取出最大的众数
# modes = counts.index[:1]
# print(modes)
# # [65, 67, 64, 69, 62]
# # 输出众数完整记录
# for mode in modes:
#     count = counts[mode]
#     # 筛选出 Avg kph 等于众数的记录
#     records = df[df['Avg kph'] == mode]
#     # 输出众数的前五条完整记录
#     print(f"Avg kph: {mode}, count: {count}")
#     print(records.head(10))
#     print()
'''
---------------------------------------------
车流量较大值的前十个值：
datetime (Veh/15 Minutes)  Avg kph  Flow
        2020-02-04 06:45       65   197
        2020-01-12 06:45       65   195
        2020-02-22 06:40       65   191
        2020-02-17 06:45       64   188
        2020-01-06 07:00       67   186
        2020-02-10 06:35       67   185
        2020-02-24 06:35       65   183
        2020-03-15 06:45       67   183
        2020-01-13 06:50       67   181
        2020-03-04 07:00       64   181
鉴于我的数据来源于伦敦到罗切斯特的A2国道，且没有查询到相关道路交通容量信息，仅知道该路段为双向行驶的限速为50英里/小时（80公里/小时）的道路，通过车流量较大的前十个值（取正常通行情况）我们可以粗略得出一个大概的道路容量253
拥堵率为每一个时段车流量/道路容量*100%，预测的拥堵率也根据预测车流量/道路容量*100%计算。
'''
# 估计道路容量
# 80/65=x/313
# 80/65=x/195
# 80/65=x/191
# 80/64=x/188
a = [[65,197],[65,195],[65,191],[64,188],[67,186],[67,185],[65,183],[67,183],[67,181],[64,181]]
# x = 80/a[0][0]*a[0][1]
x = 0
for item in a:
    x += 80/item[0]*item[1]
X = x/9
print(X)
#a[9]:  253.51457456308196


print('---------------------------------------------\n判断是否存在异常数据：')
#判断是否有异常数据，避免影响模型训练。说明：此时道路上有车流，通畅率0%
if df[(df['Avg kph'] == 0) & (df['Flow'] != 0)].empty:
    print("总数据记录为："+len(df).__str__()+"\n异常数据为空")
else:
    print("存在异常数据")
    print(df[(df['Avg kph'] == 0) & (df['Flow'] != 0)])
    # 将 Avg kph 列中的值为 0 的记录替换为 NaN
    df.loc[df['Avg kph'] == 0, 'Avg kph'] = pd.NaT

    # 使用前向和后向填充缺失值
    df['Avg kph'] = df['Avg kph'].fillna(method='ffill').fillna(method='bfill')

    # # 计算 Flow 列
    # df['Flow'] = df['Flow'].interpolate(method='linear')

    # df.loc[df['Avg kph'] == 0, 'Flow'] = 0
# 输出结果到webapp/data/，方便后续web程序调用
# print(df.head())
df.to_csv('../Traffic_Data/output_test_data.csv', index=False, header=['datetime (Veh/15 Minutes)', 'Flow', 'Avg kph'])
df.to_csv('../../webapp/data/output_test_data.csv', index=False, header=['datetime (Veh/15 Minutes)', 'Flow', 'Avg kph'])