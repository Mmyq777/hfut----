import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

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
print(f"原始训练集 y=1 占比: {y_train.mean():.2%}")

# ==================== 2. SMOTE 过采样 ====================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE 后训练集: {X_train_smote.shape}, y=1 占比: {y_train_smote.mean():.2%}")

# 计算 scale_pos_weight（负样本数 / 正样本数）
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ==================== 3. 三个模型对比 ====================

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{'='*60}")
    print(f"【{name}】")
    print(f"{'='*60}")
    print(f"准确率  (Accuracy):  {accuracy_score(y_test, y_pred):.4f}")
    print(f"精确率  (Precision): {precision_score(y_test, y_pred):.4f}")
    print(f"召回率  (Recall):    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 分数 (F1):        {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:             {roc_auc_score(y_test, y_prob):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['no', 'yes']))
    return y_pred, y_prob

# 模型 1：默认 XGBoost
xgb_default = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_default.fit(X_train, y_train)
y_pred_1, y_prob_1 = evaluate(xgb_default, X_test, y_test, 'XGBoost 默认')

# 模型 2：XGBoost + scale_pos_weight
xgb_weighted = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42, eval_metric='logloss'
)
xgb_weighted.fit(X_train, y_train)
y_pred_2, y_prob_2 = evaluate(xgb_weighted, X_test, y_test, 'XGBoost + scale_pos_weight')

# 模型 3：XGBoost + SMOTE
xgb_smote = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_smote.fit(X_train_smote, y_train_smote)
y_pred_3, y_prob_3 = evaluate(xgb_smote, X_test, y_test, 'XGBoost + SMOTE')

# ==================== 4. 阈值调优（在 SMOTE 模型上） ====================
print(f"\n{'='*60}")
print("【SMOTE 模型 — 阈值调优】")
print(f"{'='*60}")
print(f"{'阈值':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 48)

best_f1, best_t = 0, 0.5
for t in np.arange(0.2, 0.55, 0.05):
    y_pred_t = (y_prob_3 >= t).astype(int)
    acc = accuracy_score(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t)
    rec = recall_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)
    print(f"{t:<8.2f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"\n最佳阈值: {best_t:.2f} (F1={best_f1:.4f})")

# ==================== 5. 画图 ====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC 对比
for name, y_prob, style in [
    ('默认', y_prob_1, '--'),
    ('scale_pos_weight', y_prob_2, '-.'),
    ('SMOTE', y_prob_3, '-')
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, style, label=f'{name} (AUC={auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k:', label='随机猜测')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC 曲线对比')
axes[0].legend()

# 混淆矩阵（SMOTE）
cm = confusion_matrix(y_test, y_pred_3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['no', 'yes'], yticklabels=['no', 'yes'], ax=axes[1])
axes[1].set_title('XGBoost + SMOTE 混淆矩阵')
axes[1].set_xlabel('预测值')
axes[1].set_ylabel('真实值')

# 三模型 Recall / F1 对比
models = ['默认', 'scale_pos_weight', 'SMOTE']
recalls = [recall_score(y_test, y_pred_1),
           recall_score(y_test, y_pred_2),
           recall_score(y_test, y_pred_3)]
f1s = [f1_score(y_test, y_pred_1),
       f1_score(y_test, y_pred_2),
       f1_score(y_test, y_pred_3)]

x_pos = np.arange(len(models))
width = 0.35
bars1 = axes[2].bar(x_pos - width/2, recalls, width, label='Recall', color='steelblue')
bars2 = axes[2].bar(x_pos + width/2, f1s, width, label='F1', color='coral')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(models)
axes[2].set_ylabel('分数')
axes[2].set_title('三模型 Recall / F1 对比')
axes[2].legend()
for bar in bars1:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', fontsize=9)
for bar in bars2:
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=150)
plt.show()

# ==================== 6. 特征重要性 Top 15 ====================
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = list(num_cols) + list(cat_feature_names)

importances = pd.DataFrame({
    'feature': all_feature_names,
    'importance': xgb_smote.feature_importances_
}).sort_values('importance', ascending=False).head(15)

print("\n【特征重要性 Top 15（XGBoost + SMOTE）】")
for _, row in importances.iterrows():
    print(f"  {row['feature']:<35s} {row['importance']:.4f}")