import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

# ==================== 2. 默认随机森林 ====================
print("\n" + "=" * 60)
print("【随机森林 — 默认参数】")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"准确率  (Accuracy):  {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率  (Precision): {precision_score(y_test, y_pred):.4f}")
print(f"召回率  (Recall):    {recall_score(y_test, y_pred):.4f}")
print(f"F1 分数 (F1):        {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_test, y_prob):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['no', 'yes']))

# ==================== 3. class_weight='balanced' ====================
print("=" * 60)
print("【随机森林 — class_weight='balanced'】")
print("=" * 60)

rf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     random_state=42, n_jobs=-1)
rf_balanced.fit(X_train, y_train)
y_pred_b = rf_balanced.predict(X_test)
y_prob_b = rf_balanced.predict_proba(X_test)[:, 1]

print(f"准确率  (Accuracy):  {accuracy_score(y_test, y_pred_b):.4f}")
print(f"精确率  (Precision): {precision_score(y_test, y_pred_b):.4f}")
print(f"召回率  (Recall):    {recall_score(y_test, y_pred_b):.4f}")
print(f"F1 分数 (F1):        {f1_score(y_test, y_pred_b):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_test, y_prob_b):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_b, target_names=['no', 'yes']))

# ==================== 4. 画图 ====================
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
results = []

for t in thresholds:
    y_pred_t = (y_prob_b >= t).astype(int)
    results.append({
        '阈值': t,
        'Accuracy': accuracy_score(y_test, y_pred_t),
        'Precision': precision_score(y_test, y_pred_t),
        'Recall': recall_score(y_test, y_pred_t),
        'F1': f1_score(y_test, y_pred_t)
    })

print("\n【不同阈值表现】")
print(f"{'阈值':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 48)
for r in results:
    print(f"{r['阈值']:<8} {r['Accuracy']:<10.4f} {r['Precision']:<10.4f} {r['Recall']:<10.4f} {r['F1']:<10.4f}")

# 画阈值对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Precision / Recall / F1 vs 阈值
t_list = [r['阈值'] for r in results]
axes[0].plot(t_list, [r['Precision'] for r in results], 'o-', label='Precision')
axes[0].plot(t_list, [r['Recall'] for r in results], 's-', label='Recall')
axes[0].plot(t_list, [r['F1'] for r in results], '^-', label='F1')
axes[0].set_xlabel('阈值')
axes[0].set_title('Precision / Recall / F1 随阈值变化')
axes[0].legend()
axes[0].invert_xaxis()  # 阈值从大到小，方便看

# ROC 曲线（标注不同阈值的位置）
fpr_b, tpr_b, thresholds_roc = roc_curve(y_test, y_prob_b)
axes[1].plot(fpr_b, tpr_b, label=f'Random Forest (AUC={roc_auc_score(y_test, y_prob_b):.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='随机猜测')
for t in [0.3, 0.4, 0.5]:
    y_pred_t = (y_prob_b >= t).astype(int)
    fpr_t = 1 - recall_score(y_test, y_pred_t, pos_label=0)  # specificity 的补集
    rec_t = recall_score(y_test, y_pred_t)
    axes[1].scatter(fpr_t, rec_t, s=100, label=f'阈值={t}', zorder=5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC 曲线（含不同阈值）')
axes[1].legend()

plt.tight_layout()
plt.savefig('rf_threshold_tuning.png', dpi=150)
plt.show()

# ==================== 5. 特征重要性 Top 15 ====================
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = list(num_cols) + list(cat_feature_names)

importances = pd.DataFrame({
    'feature': all_feature_names,
    'importance': rf_balanced.feature_importances_
}).sort_values('importance', ascending=False).head(15)

print("\n【特征重要性 Top 15（随机森林）】")
for _, row in importances.iterrows():
    print(f"  {row['feature']:<35s} {row['importance']:.4f}")