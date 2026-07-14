# 🏦 银行定期存款预测 — 机器学习大作业

基于葡萄牙银行电话营销数据，构建机器学习模型预测客户是否会订购定期存款。

## 📊 数据集

| 项目 | 说明 |
|------|------|
| 来源 | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| 样本数 | 45,211 |
| 特征数 | 16（剔除 duration 后 15 个） |
| 目标变量 | `y` — 是否订购定期存款（yes/no，正类占比 11.70%） |
| 特点 | 类别不平衡、含 unknown 缺失值、duration 为数据泄露特征需剔除 |

### 输入特征

| 类型 | 特征 | 说明 |
|------|------|------|
| 数值 | `age` | 年龄 |
| 数值 | `balance` | 账户年均余额（欧元） |
| 数值 | `campaign` | 本次营销活动联系次数 |
| 数值 | `pdays` | 距上次营销联系天数（-1 = 从未联系） |
| 数值 | `previous` | 之前营销活动联系次数 |
| 分类 | `job` | 职业（12 种） |
| 分类 | `marital` | 婚姻状况 |
| 分类 | `education` | 教育程度 |
| 分类 | `default` | 是否有信贷违约 |
| 分类 | `housing` | 是否有住房贷款 |
| 分类 | `loan` | 是否有个人贷款 |
| 分类 | `contact` | 联系方式（手机/固话） |
| 分类 | `day_of_week` | 最后联系星期（1-5） |
| 分类 | `month` | 最后联系月份 |
| 分类 | `poutcome` | 上次营销结果 |

> ⚠️ `duration`（通话时长）在训练时已剔除，因为实际预测时电话未拨出、无法获知。

---

## 🏗️ 项目结构

```
├── download.py                # 从 UCI 下载数据集
├── eda.py                     # 探索性数据分析
├── preprocess.py              # 数据预处理
├── train_logistic.py          # 逻辑回归基线
├── random_forest_train.py     # 随机森林 + 阈值调优
├── xgboost_train.py           # XGBoost 三种不平衡处理方案
├── tune_xgboost.py            # 超参数调优 + 模型保存
├── predict.py                 # 加载模型对新数据预测
├── front/
│   ├── streamlit_app.py       # Streamlit 前端交互界面
│   └── requirements.txt       # 前端依赖
├── xgboost_best_model.pkl     # 最优模型
├── preprocessor.pkl           # 预处理器
└── README.md
```

---

## 🚀 快速开始

### 环境要求

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib streamlit
```

### 训练模型

```bash
# 1. 下载数据
python download.py

# 2. 数据探索
python eda.py

# 3. 超参数调优 + 保存最优模型（核心，跑这个就够了）
python tune_xgboost.py
```

### 预测

```bash
# 将新数据准备好后
python predict.py
```

### 启动前端界面

```bash
cd front
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501`，填入客户信息即可实时预测。

---

## 📈 模型性能对比

| 模型 | Accuracy | Recall | F1 | ROC-AUC |
|------|:--------:|:------:|:--:|:-------:|
| 逻辑回归 (class_weight=balanced) | 76.19% | **63.33%** | 38.36% | 0.7737 |
| 随机森林 (阈值=0.3) | 87.79% | 42.53% | 44.91% | 0.7835 |
| XGBoost + scale_pos_weight | 82.09% | 58.60% | 43.36% | 0.7763 |
| XGBoost + SMOTE (阈值=0.25) | 86.42% | 51.32% | 46.93% | 0.7872 |
| **🏆 调优 XGBoost (阈值=0.40)** | 77.21% | 66.92% | 40.72% | **0.7915** |

最优模型：**调优 XGBoost**，AUC 最高（0.7915），Recall 可达 66.92%（阈值 0.40）。

---

## 🔧 技术栈

- **语言**: Python 3.10
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **模型**: Scikit-learn, XGBoost
- **不平衡处理**: SMOTE (imbalanced-learn), scale_pos_weight
- **调优**: RandomizedSearchCV
- **前端**: Streamlit
