import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==================== 读取数据 ====================
df = pd.read_csv('bank_marketing.csv')

# ==================== 1. 剔除 duration（预测时未知，数据泄露） ====================
df = df.drop(columns=['duration'])

# ==================== 2. 查看并处理 unknown ====================
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('y')

print("各列 unknown 占比：")
for col in cat_cols:
    unknown_count = (df[col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"  {col}: {unknown_count} 个 ({unknown_count/len(df)*100:.1f}%)")

# 用众数替换 unknown
for col in cat_cols:
    if (df[col] == 'unknown').sum() > 0:
        mode_val = df[col][df[col] != 'unknown'].mode()[0]
        df[col] = df[col].replace('unknown', mode_val)

# ==================== 3. 目标变量转 0/1 ====================
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# ==================== 4. 分离 X 和 y ====================
X = df.drop(columns=['y'])
y = df['y']
X['day_of_week'] = X['day_of_week'].astype(str)
# ==================== 5. 区分列类型 ====================
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

print(f"\n数值列 ({len(num_cols)}): {num_cols}")
print(f"分类列 ({len(cat_cols)}): {cat_cols}")

# ==================== 6. 预处理器：数值标准化 + 分类OneHot ====================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

X_processed = preprocessor.fit_transform(X)
print(f"\n处理后的特征维度: {X_processed.shape}")

# ==================== 7. 划分训练集/测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集 y=1 占比: {y_train.mean():.2%}")
print(f"测试集 y=1 占比: {y_test.mean():.2%}")
print("\n预处理完成！")
