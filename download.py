from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# 合并为完整数据框
df = pd.concat([X, y], axis=1)

# 保存到本地 (后续步骤都从这个CSV读取，不用重复下载)
df.to_csv('bank_marketing.csv', index=False)

# metadata
print(bank_marketing.metadata)

# variable information
print(bank_marketing.variables)

print("\n已保存 bank_marketing.csv，形状:", df.shape)