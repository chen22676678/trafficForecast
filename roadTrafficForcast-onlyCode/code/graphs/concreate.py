import pandas as pd
'''
合并Flow、LSTM、GRU数据
'''
# 读取两个 CSV 文件并转换成 DataFrame 对象
df1 = pd.read_csv('../../webapp/data/output_test_data_expand.csv')
df2 = pd.read_csv('result.csv')
# 重命名 df1 中的 Flow 列为 True
df1 = df1.rename(columns={'Flow': 'True'})

# 指定要根据哪些列进行合并，将两个 DataFrame 对象合并成一个新的 DataFrame 对象
merged_df = pd.merge(df1[['datetime (Veh/5 Minutes)', 'True']],
                     df2[['datetime (Veh/5 Minutes)', 'LSTM', 'GRU']],
                     on='datetime (Veh/5 Minutes)')

# 将合并后的结果保存到新的 CSV 文件中
merged_df.to_csv('merged_file.csv', index=False)
