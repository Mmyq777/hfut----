import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, accuracy_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 预处理 ====================
print("正在加载和预处理数据...")
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

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ==================== 2. 定义搜索空间 ====================
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2],
}

# ==================== 3. RandomizedSearchCV ====================
print("\n正在随机搜索超参数（总共 50 组，大约需要几分钟）...")

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print(f"\n最佳 F1 (CV): {search.best_score_:.4f}")
print(f"最佳参数:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

# ==================== 4. 用最佳参数重新训练 ====================
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(f"\n{'='*60}")
print("【调优后 XGBoost — 测试集表现】")
print(f"{'='*60}")
print(f"准确率  (Accuracy):  {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率  (Precision): {precision_score(y_test, y_pred):.4f}")
print(f"召回率  (Recall):    {recall_score(y_test, y_pred):.4f}")
print(f"F1 分数 (F1):        {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_test, y_prob):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['no', 'yes']))

# ==================== 5. 阈值调优 ====================
print(f"\n{'='*60}")
print("【调优模型 — 阈值调整】")
print(f"{'='*60}")
print(f"{'阈值':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 48)

best_f1, best_t = 0, 0.5
best_recall, best_recall_t = 0, 0.5

for t in np.arange(0.2, 0.55, 0.05):
    y_pred_t = (y_prob >= t).astype(int)
    acc = accuracy_score(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t)
    rec = recall_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)
    print(f"{t:<8.2f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    if f1 > best_f1:
        best_f1, best_t = f1, t
    if rec > best_recall:
        best_recall, best_recall_t = rec, t

print(f"\n最佳 F1 阈值: {best_t:.2f} (F1={best_f1:.4f})")
print(f"最高 Recall 阈值: {best_recall_t:.2f} (Recall={best_recall:.4f})")

# ==================== 6. 画图 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 混淆矩阵（最佳 F1 阈值）
y_pred_best = (y_prob >= best_t).astype(int)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['no', 'yes'], yticklabels=['no', 'yes'], ax=axes[0])
axes[0].set_title(f'调优 XGBoost — 混淆矩阵 (阈值={best_t:.2f})')
axes[0].set_xlabel('预测值')
axes[0].set_ylabel('真实值')

# ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, label=f'调优 XGBoost (AUC={roc_auc_score(y_test, y_prob):.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='随机猜测')
for t_label, rec_val in [('F1最优', best_t), ('Recall最高', best_recall_t)]:
    y_pred_t = (y_prob >= rec_val).astype(int)
    fpr_t = 1 - recall_score(y_test, y_pred_t, pos_label=0)
    axes[1].scatter(fpr_t, recall_score(y_test, y_pred_t), s=100,
                    label=f'{t_label} ({rec_val:.2f})', zorder=5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC 曲线')
axes[1].legend()

plt.tight_layout()
plt.savefig('xgboost_tuned.png', dpi=150)
plt.show()

# ==================== 7. 汇总对比 ====================
print(f"\n{'='*60}")
print("【最终模型对比汇总】")
print(f"{'='*60}")
print(f"{'模型':<30} {'Accuracy':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
print("-" * 70)
# 这里把之前几个模型的最好结果写死，方便最终对比
print(f"{'逻辑回归 Balanced':<30} {'0.7619':<10} {'0.6333':<10} {'0.3836':<10} {'0.7737':<10}")
print(f"{'随机森林 阈值=0.3':<30} {'0.8779':<10} {'0.4253':<10} {'0.4491':<10} {'0.7835':<10}")
print(f"{'XGBoost scale_pos_weight':<30} {'0.8209':<10} {'0.5860':<10} {'0.4336':<10} {'0.7763':<10}")
print(f"{'XGBoost SMOTE 阈值=0.25':<30} {'0.8642':<10} {'0.5132':<10} {'0.4693':<10} {'0.7872':<10}")
print(f"{'XGBoost 调优 阈值={:.2f}'.format(best_t):<30} "
      f"{accuracy_score(y_test, y_pred_best):<10.4f} {recall_score(y_test, y_pred_best):<10.4f} "
      f"{f1_score(y_test, y_pred_best):<10.4f} {roc_auc_score(y_test, y_prob):<10.4f}")
#保存最优模型
import joblib


joblib.dump(best_model, 'xgboost_best_model.pkl')

# 保存预处理器（预测新数据时也要用同样的预处理）
joblib.dump(preprocessor, 'preprocessor.pkl')

print("模型已保存: xgboost_best_model.pkl")
print("预处理器已保存: preprocessor.pkl")