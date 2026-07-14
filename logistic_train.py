#逻辑回归训练
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, accuracy_score)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 读取并预处理 ====================
df = pd.read_csv('bank_marketing.csv')
df = df.drop(columns=['duration'])
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('y')

for col in cat_cols:
    if (df[col] == 'unknown').sum() > 0:
        mode_val = df[col][df[col] != 'unknown'].mode()[0]
        df[col] = df[col].replace('unknown', mode_val)

df['y'] = df['y'].map({'yes': 1, 'no': 0})
X = df.drop(columns=['y'])
y = df['y']
X['day_of_week'] = X['day_of_week'].astype(str)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, stratify=y, random_state=42
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集 y=1 占比: {y_train.mean():.2%}")

# ==================== 2. 默认逻辑回归 ====================
print("\n" + "=" * 60)
print("【逻辑回归 — 默认参数】")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

print(f"准确率  (Accuracy):  {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率  (Precision): {precision_score(y_test, y_pred):.4f}")
print(f"召回率  (Recall):    {recall_score(y_test, y_pred):.4f}")
print(f"F1 分数 (F1):        {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_test, y_prob):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['no', 'yes']))

# ==================== 3. class_weight='balanced' ====================
print("=" * 60)
print("【逻辑回归 — class_weight='balanced'】")
print("=" * 60)

lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_balanced.fit(X_train, y_train)
y_pred_b = lr_balanced.predict(X_test)
y_prob_b = lr_balanced.predict_proba(X_test)[:, 1]

print(f"准确率  (Accuracy):  {accuracy_score(y_test, y_pred_b):.4f}")
print(f"精确率  (Precision): {precision_score(y_test, y_pred_b):.4f}")
print(f"召回率  (Recall):    {recall_score(y_test, y_pred_b):.4f}")
print(f"F1 分数 (F1):        {f1_score(y_test, y_pred_b):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_test, y_prob_b):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_b, target_names=['no', 'yes']))

# ==================== 4. 画图 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_b)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['no', 'yes'], yticklabels=['no', 'yes'], ax=axes[0])
axes[0].set_title('混淆矩阵 (class_weight=balanced)')
axes[0].set_xlabel('预测值')
axes[0].set_ylabel('真实值')

# ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_b)
axes[1].plot(fpr, tpr, label=f'默认 (AUC={roc_auc_score(y_test, y_prob):.3f})')
axes[1].plot(fpr_b, tpr_b, label=f'Balanced (AUC={roc_auc_score(y_test, y_prob_b):.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='随机猜测')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC 曲线')
axes[1].legend()

plt.tight_layout()
plt.savefig('logistic_results.png', dpi=150)
plt.show()

# ==================== 5. 特征重要性 ====================
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = list(num_cols) + list(cat_feature_names)

coefs = pd.DataFrame({'feature': all_feature_names, 'coefficient': lr_balanced.coef_[0]})
coefs['abs'] = coefs['coefficient'].abs()
top10 = coefs.sort_values('abs', ascending=False).head(10)

print("\n【特征重要性 Top 10（系数绝对值）】")
for _, row in top10.iterrows():
    direction = "→ 促进订阅" if row['coefficient'] > 0 else "→ 抑制订阅"
    print(f"  {row['feature']:<35s} {row['coefficient']:>8.4f}  {direction}")