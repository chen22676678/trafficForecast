import pandas as pd
import matplotlib.pyplot as plt
'''
判断Flow and Avg Kph字段值是否具有相关性，能否用做lstm训练的特征输入

完成
'''
data = pd.read_csv('../Traffic_Data/output_test_data_expand.csv', header=0, parse_dates=True)
# data = pd.read_csv('../Traffic_Data/DailyStandard_Report_1_8978_01_04_2019_31_05_2019.csv', header=0, parse_dates=True)
corr = data['Flow'].corr(data['Avg kph'])
# corr = data['Total Volume'].corr(data['0 - 520 cm'])

print("Correlation between Flow and Avg Kph: {:.2f}".format(corr))

fig = plt.figure(figsize=(20, 10))
plt.scatter(data['Avg kph'], data['Flow'], alpha=0.5)
plt.xlabel('Avg kph')
plt.ylabel('Flow')
plt.show()
fig.savefig("../graphs/Flow-vs-Avg_kph.png")
