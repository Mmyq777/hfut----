import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 用python

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 读取数据 ====================
df = pd.read_csv('bank_marketing.csv')

# ==================== 1. 数据基本信息 ====================
print("=" * 50)
print("数据形状:", df.shape)
print("=" * 50)

print("\n列名及类型:")
print(df.dtypes)

print("\n前5行:")
print(df.head())

print("\n目标变量 y 分布:")
print(df['y'].value_counts())
print(f"\n占比:")
print(df['y'].value_counts(normalize=True).mul(100).round(2).to_string())

# ==================== 2. 缺失值检查 ====================
print("\n" + "=" * 50)
print("缺失值统计:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ==================== 3. 数值特征统计 ====================
print("\n" + "=" * 50)
print("数值特征统计:")
print(df.describe())

# ==================== 4. 画图 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 目标变量分布
df['y'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['steelblue', 'coral'])
axes[0, 0].set_title('目标变量 y 分布')
axes[0, 0].set_xlabel('是否订购')
axes[0, 0].set_ylabel('数量')

# age 分布
axes[0, 1].hist(df['age'], bins=30, color='steelblue', edgecolor='white')
axes[0, 1].set_title('年龄分布')
axes[0, 1].set_xlabel('年龄')
axes[0, 1].set_ylabel('频数')

# duration 分布
axes[1, 0].hist(df['duration'], bins=50, color='seagreen', edgecolor='white')
axes[1, 0].set_title('通话时长分布')
axes[1, 0].set_xlabel('通话时长(秒)')
axes[1, 0].set_ylabel('频数')

# balance 分布
axes[1, 1].hist(df['balance'], bins=50, color='purple', edgecolor='white')
axes[1, 1].set_title('账户余额分布')
axes[1, 1].set_xlabel('余额(欧元)')
axes[1, 1].set_ylabel('频数')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150)
plt.show()