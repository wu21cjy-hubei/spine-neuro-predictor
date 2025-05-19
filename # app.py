# app.py
import streamlit as st
import pandas as pd
import joblib

# 加载模型
model = joblib.load("/Users/jingyicai/Desktop/xgb_neuro_model.pkl")

st.title("神经症状预测工具")
st.markdown("基于 MRI 与实验室特征的 XGBoost 模型")

# 用户输入界面
bone_abscess = st.selectbox("骨内小脓肿（0=无, 1=有）", [0, 1])
t2_signal = st.selectbox("T2WI信号强度（1=低, 2=高, 3=混合）", [1, 2, 3])
endplate_line = st.selectbox("终板高信号（0=无, 1=有）", [0, 1])
crp = st.number_input("CRP（mg/L）", value=10.0)
neutrophil_percent = st.number_input("中性粒细胞占比（%）", value=65.0)

# 生成特征 DataFrame
input_data = pd.DataFrame([[bone_abscess, t2_signal, endplate_line, crp, neutrophil_percent]],
                          columns=['骨内小脓肿（无0，有1）',
                                   '受累椎体T2WI信号强度（1低信号，2高信号，3混合',
                                   '椎间盘周围环形高信号',
                                   'CRP',
                                   '中性粒细胞占比'])

# 预测
if st.button("预测神经症状风险"):
    pred_prob = model.predict_proba(input_data)[0][1]
    st.success(f"预测患有神经症状的概率为：{pred_prob:.2%}")