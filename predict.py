import pandas as pd
import joblib

# 1. 加载保存的模型和预处理器
model = joblib.load('xgboost_best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# 2. 读取新数据（格式要和训练时完全一样）
df_new = pd.read_csv('new_data.csv')
df_new = df_new.drop(columns=['duration'])
df_new['day_of_week'] = df_new['day_of_week'].astype(str)

# 3. 预处理 + 预测
X_new = preprocessor.transform(df_new)
y_prob = model.predict_proba(X_new)[:, 1]        # 概率
y_pred = (y_prob >= 0.40).astype(int)              # 类别（用0.4阈值）

print("预测完成，结果:")
print(pd.Series(y_pred).value_counts())