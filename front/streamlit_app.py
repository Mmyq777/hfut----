import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="银行定期存款预测",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 加载模型 ====================
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'xgboost_best_model.pkl')
    preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'preprocessor.pkl')
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_model()

# ==================== 自定义 CSS ====================
st.markdown("""
<style>
    /* 全局字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 主标题 */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        text-align: center;
        color: #5f6368;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* 卡片样式 */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #e8eaed;
        margin-bottom: 1rem;
    }
    .card-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a73e8;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #e8f0fe;
    }

    /* 预测按钮 */
    .stButton > button {
        width: 100%;
        height: 3.2rem;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%) !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26,115,232,0.35) !important;
    }

    /* 结果卡片 */
    .result-card-yes {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #4caf50;
    }
    .result-card-no {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #1a73e8;
    }
    .result-label {
        font-size: 1rem;
        color: #5f6368;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-prob {
        font-size: 1.2rem;
        color: #5f6368;
    }

    /* 概率条 */
    .prob-bar-container {
        background: #e8eaed;
        border-radius: 10px;
        height: 12px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease;
    }

    /* 提示框 */
    .tip-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #5d4037;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 页面标题 ====================
st.markdown('<div class="main-title">🏦 银行定期存款预测系统</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Bank Term Deposit Subscription Predictor — 基于 XGBoost 机器学习模型</div>', unsafe_allow_html=True)

# ==================== 输入区域 ====================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">👤 客户基本信息</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input('年龄', min_value=18, max_value=95, value=35, step=1, help='客户年龄')
        job = st.selectbox('职业', [
            'admin.', 'blue-collar', 'entrepreneur', 'housemaid',
            'management', 'retired', 'self-employed', 'services',
            'student', 'technician', 'unemployed'
        ], help='客户职业类型')
        marital = st.selectbox('婚姻状况', ['married', 'single', 'divorced'],
                               help='已婚 / 单身 / 离异或丧偶')
        education = st.selectbox('教育程度', [
            'primary', 'secondary', 'tertiary'
        ], help='小学 / 中学 / 大学')
        balance = st.number_input('账户年均余额 (欧元)', min_value=-10000, max_value=100000,
                                  value=1500, step=100, help='账户平均年余额，可为负')

    with c2:
        default = st.selectbox('是否有信贷违约', ['no', 'yes'],
                               help='是否有信用卡或个人贷款违约记录')
        housing = st.selectbox('是否有住房贷款', ['no', 'yes'])
        loan = st.selectbox('是否有个人贷款', ['no', 'yes'])
        # 空行占位，保持对齐
        st.markdown('<br>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📞 营销联系信息</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        contact = st.selectbox('联系方式', ['cellular', 'telephone'],
                               help='手机 / 固定电话')
        month = st.selectbox('最后联系月份', [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ])
        day_of_week = st.selectbox('最后联系星期', ['1', '2', '3', '4', '5'],
                                format_func=lambda x: {'1': '周一', '2': '周二', '3': '周三',
                                                       '4': '周四', '5': '周五'}.get(x, x))
        campaign = st.number_input('本次营销联系次数', min_value=1, max_value=50,
                                   value=2, step=1)

    with c4:
        poutcome = st.selectbox('上次营销结果', ['nonexistent', 'failure', 'success'],
                                help='不存在(首次联系) / 失败 / 成功')

        # 是否曾被联系过
        prev_contacted = st.checkbox('曾被之前的营销活动联系过', value=False)
        if prev_contacted:
            pdays = st.number_input('距上次联系天数', min_value=0, max_value=999,
                                    value=90, step=1)
            previous = st.number_input('之前营销联系次数', min_value=0, max_value=50,
                                       value=1, step=1)
        else:
            pdays = -1
            previous = 0
            st.caption('未联系过 → pdays = -1, previous = 0')

    st.markdown('</div>', unsafe_allow_html=True)

# ==================== 预测按钮 ====================
st.markdown('<br>', unsafe_allow_html=True)
predict_btn = st.button('🔮 开始预测')

# ==================== 预测逻辑 ====================
if predict_btn:
    # 构建输入 DataFrame
    input_data = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day_of_week': day_of_week,
        'month': month,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
    }])

    # 预处理
    input_data['day_of_week'] = input_data['day_of_week'].astype(str)
    X_input = preprocessor.transform(input_data)

    # 预测
    prob = model.predict_proba(X_input)[0, 1]
    pred = int(prob >= 0.40)

    # ==================== 结果展示 ====================
    st.markdown('<br>', unsafe_allow_html=True)

    # 三列布局展示结果
    res_col1, res_col2, res_col3 = st.columns([1, 2, 1])

    with res_col2:
        if pred == 1:
            st.markdown(f"""
            <div class="result-card-yes">
                <div class="result-label">📊 预测结果</div>
                <div class="result-value" style="color: #2e7d32;">✅ 会订购定期存款</div>
                <div class="result-prob">置信度: {prob:.1%}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: {prob*100}%; background: linear-gradient(90deg, #66bb6a, #2e7d32);"></div>
                </div>
                <div style="font-size: 0.85rem; color: #5f6368; margin-top: 0.5rem;">
                    该客户有 <b>{prob:.0%}</b> 的概率订购定期存款
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-no">
                <div class="result-label">📊 预测结果</div>
                <div class="result-value" style="color: #1565c0;">ℹ️ 不太可能订购定期存款</div>
                <div class="result-prob">订购概率: {prob:.1%}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: {prob*100}%; background: linear-gradient(90deg, #42a5f5, #1565c0);"></div>
                </div>
                <div style="font-size: 0.85rem; color: #5f6368; margin-top: 0.5rem;">
                    该客户仅有 <b>{prob:.0%}</b> 的概率订购定期存款
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 关键影响因素提示
    st.markdown('<br>', unsafe_allow_html=True)
    with st.expander('📋 查看输入摘要', expanded=False):
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric('年龄', f'{age} 岁')
            st.metric('职业', job)
            st.metric('教育程度', education)
            st.metric('婚姻状况', marital)
            st.metric('账户余额', f'€{balance:,}')
        with info_col2:
            st.metric('联系方式', contact)
            st.metric('联系月份', month)
            st.metric('联系星期', day_of_week)
            st.metric('本次联系次数', campaign)
            st.metric('距上次联系', f'{pdays} 天' if pdays >= 0 else '未联系过')
        with info_col3:
            st.metric('信贷违约', '是' if default == 'yes' else '否')
            st.metric('住房贷款', '是' if housing == 'yes' else '否')
            st.metric('个人贷款', '是' if loan == 'yes' else '否')
            st.metric('上次营销结果', poutcome)
            st.metric('之前联系次数', previous)

    # 提示
    st.markdown("""
    <div class="tip-box">
        💡 <b>提示</b>：模型使用 <b>0.40 阈值</b>（偏向高召回率），实际使用中可根据业务需求调整。
        阈值越低 → 找出更多潜在客户但误报更多；阈值越高 → 预测更精准但可能漏掉客户。
    </div>
    """, unsafe_allow_html=True)
