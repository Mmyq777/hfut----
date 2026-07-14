import pandas as pd
import joblib

# 1. 加载保存的模型和预处理器
model = joblib.load('xgboost_best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# 2. 读取新数据
df_new = pd.read_csv('new_data.csv')

# 3. 与训练时完全一致的预处理
df_new = df_new.drop(columns=['duration'], errors='ignore')       # 剔除 duration

cat_cols = df_new.select_dtypes(include='object').columns.tolist()
if 'y' in cat_cols:
    cat_cols.remove('y')
    df_new = df_new.drop(columns=['y'])                           # 去掉标签列

for col in cat_cols:
    if (df_new[col] == 'unknown').sum() > 0:
        mode_val = df_new[col][df_new[col] != 'unknown'].mode()[0]
        df_new[col] = df_new[col].replace('unknown', mode_val)

df_new['day_of_week'] = df_new['day_of_week'].astype(str)

# 4. 预处理 + 预测
X_new = preprocessor.transform(df_new)
y_prob = model.predict_proba(X_new)[:, 1]          # 概率
y_pred = (y_prob >= 0.40).astype(int)               # 类别（阈值0.4，Recall≈67%）

print("预测完成！")
print(f"总样本: {len(y_pred)}")
print(f"预测为 yes (1): {y_pred.sum()} ({y_pred.mean():.2%})")
print(f"预测为 no  (0): {(1-y_pred).sum()} ({(1-y_pred).mean():.2%})")