#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.svm import SVC

# 解决matplotlib绘图潜在后端问题
plt.switch_backend('Agg')

# -------------------------- 1. 路径配置（改为训练代码的绝对路径） --------------------------
# 替换为你训练代码中模型/scaler/特征名/阈值的实际保存路径
MODEL_PATH = "svm_best_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
THRESHOLD_PATH = "best_thresholds.pkl"

# -------------------------- 2. 加载模型、标准化器、特征名、最佳阈值 --------------------------
try:
    # 加载线性SVM模型并验证
    model = joblib.load(MODEL_PATH)
    if not (isinstance(model, SVC) and model.kernel == "linear"):
        st.error("加载的模型不是线性SVM，请检查模型文件路径！")
        st.stop()

    # 加载标准化器
    scaler = joblib.load(SCALER_PATH)

    # 加载训练时的特征列顺序（替换手动定义，避免顺序错位）
    FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)

    # 加载训练时的最佳阈值
    best_thresholds = joblib.load(THRESHOLD_PATH)
    svm_best_thresh = best_thresholds['svm']

    # 调试信息：验证模型/Scaler加载正确性
    st.success("模型、标准化器、特征配置加载成功！")
    st.info(f"特征顺序（与训练集一致）：{FEATURE_NAMES}")
    st.info(f"SVM最佳判定阈值：{svm_best_thresh}")
    st.info(f"模型类别映射：{model.classes_}")
    st.info(f"Scaler前5个特征均值：{scaler.mean_[:5]}")

except Exception as e:
    st.error(f"加载失败：{str(e)}，请检查文件路径是否正确！")
    st.stop()

# -------------------------- 3. 网页输入界面 --------------------------
st.title("Gastric Cancer Liver Metastasis Predictor")
st.subheader("Input Feature Values")

user_input_dict = {}

# 连续型特征输入（按训练集特征顺序生成输入框）
for feat in FEATURE_NAMES:
    if feat == "hb":
        user_input_dict[feat] = st.number_input("Hemoglobin (Hb, g/L)", min_value=30.0, max_value=200.0, value=120.0, step=0.1)
    elif feat == "tt":
        user_input_dict[feat] = st.number_input("Thrombin Time (TT, sec)", min_value=5.0, max_value=30.0, value=13.0, step=0.1)
    elif feat == "siri":
        user_input_dict[feat] = st.number_input("Systemic Inflammatory Response Index (SIRI)", min_value=0.0, max_value=10.0, value=1.0, step=0.01)
    elif feat == "afr":
        user_input_dict[feat] = st.number_input("Albumin to Fibrinogen Ratio (AFR)", min_value=0.1, max_value=20.0, value=2.5, step=0.01)
    elif feat == "cea_log":
        # CEA原始值自动转换为cea_log
        cea_original = st.number_input("CEA Original Value (ng/mL)", min_value=0.0, max_value=1000.0, value=1.47, step=0.1)
        cea_log = np.log10(cea_original + 1)
        user_input_dict[feat] = cea_log
        st.caption(f"CEA原始值={cea_original} → cea_log≈{cea_log:.6f}")
    elif feat == "lvi":
        user_input_dict[feat] = st.selectbox("Lymphovascular Invasion (LVI)", options=[0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)")
    elif feat == "t_stage":
        user_input_dict[feat] = st.selectbox("T-stage", options=[1, 2, 3, 4], format_func=lambda x: f"T{x}")
    elif feat == "n_stage":
        user_input_dict[feat] = st.selectbox("N-stage", options=[0, 1, 2, 3], format_func=lambda x: f"N{x}")

# -------------------------- 4. 构造模型输入 --------------------------
try:
    user_input_list = [user_input_dict[feat] for feat in FEATURE_NAMES]
    input_data = np.array(user_input_list).reshape(1, -1)
except KeyError as e:
    st.error(f"特征缺失：{str(e)}，请检查特征名称是否一致！")
    st.stop()

# -------------------------- 5. 预测逻辑 --------------------------
if st.button("Predict Liver Metastasis Risk"):
    # 标准化输入数据
    input_data_scaled = scaler.transform(input_data)

    # 模型预测概率
    try:
        predicted_proba = model.predict_proba(input_data_scaled)[0]
        # 使用训练时的最佳阈值判定类别（而非默认0.5）
        predicted_class = 1 if predicted_proba[1] >= svm_best_thresh else 0
    except AttributeError:
        st.warning("模型未开启概率预测，无法显示概率值！请重新训练SVM并设置probability=True")
        predicted_proba = [0, 0]
        predicted_class = model.predict(input_data_scaled)[0]

    # 展示预测结果
    st.write("### Prediction Result")
    if predicted_class == 1:
        risk_prob = predicted_proba[1] * 100 if predicted_proba[1] != 0 else "N/A"
        st.error(f"**Liver Metastasis Risk: High Risk**")
        st.write(f"Probability of Liver Metastasis: {risk_prob:.1f}%" if risk_prob != "N/A" else f"Probability of Liver Metastasis: {risk_prob}")
    else:
        risk_prob = predicted_proba[0] * 100 if predicted_proba[0] != 0 else "N/A"
        st.success(f"**Liver Metastasis Risk: Low Risk**")
        st.write(f"Probability of No Liver Metastasis: {risk_prob:.1f}%" if risk_prob != "N/A" else f"Probability of No Liver Metastasis: {risk_prob}")

    # 临床建议
    st.write("### Clinical Advice")
    advice_high = (
        "1. Complete enhanced abdominal CT/MRI within 1 month to confirm metastasis;\n"
        "2. Monitor serological indicators (Hb, CEA) every 2 weeks;\n"
        "3. Consult an oncologist for adjuvant therapy (targeted/chemotherapy);\n"
        "4. Maintain a high-protein diet to improve anemia."
    )
    advice_low = (
        "1. Follow up (abdominal ultrasound + serology) every 3 months;\n"
        "2. Avoid alcohol/spicy foods to reduce gastric irritation;\n"
        "3. Keep regular schedule & moderate exercise;\n"
        "4. Seek medical help for abdominal pain/jaundice/weight loss."
    )
    st.write(advice_high if predicted_class == 1 else advice_low)

    # -------------------------- SHAP 可视化优化 --------------------------
    st.write("### Model Interpretation (SHAP Force Plot)")
    try:
        # 替换为KernelExplainer（适配SVM，解决SHAP值计算异常）
        background_data = shap.sample(input_data_scaled, 100)
        explainer = shap.KernelExplainer(
            model=lambda x: model.predict_proba(x)[:, 1],
            data=background_data,
            feature_names=FEATURE_NAMES
        )
        shap_values = explainer.shap_values(input_data_scaled, nsamples=100)

        # 调试：展示SHAP值，验证计算正常
        st.write("#### SHAP值调试（特征贡献度）")
        shap_debug_df = pd.DataFrame(
            np.hstack([shap_values, input_data_scaled]),
            columns=[f"{f}_shap" for f in FEATURE_NAMES] + FEATURE_NAMES
        )
        st.dataframe(shap_debug_df)

        # 原生HTML渲染Force Plot（解决matplotlib渲染空白问题）
        shap_html = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values,
            features=pd.DataFrame(input_data_scaled, columns=FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            out_names="Liver Metastasis Risk",
            plot_cmap="RdBu_r",
            show=False
        )
        shap_html_str = f"<head>{shap.getjs()}</head><body>{shap_html.html()}</body>"
        components.html(shap_html_str, height=200, scrolling=True)

        # 可选：补充SHAP摘要图（辅助验证）
        st.write("#### SHAP Summary Plot (Auxiliary)")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            features=pd.DataFrame(input_data_scaled, columns=FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            plot_type="dot",
            show=False
        )
        plt.tight_layout()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"SHAP绘图失败：{str(e)}")
        st.info("可能原因：1.SVM模型未开启probability=True；2.特征值无区分度；3.SHAP版本不兼容（建议安装shap==0.42.1）")

