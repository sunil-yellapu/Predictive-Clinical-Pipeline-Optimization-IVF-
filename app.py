# IVF BLASTOCYST FORMATION PREDICTION – STREAMLIT

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


# Custom Transformer:
class IVFDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["Sperm_Concentration_log"] = np.log1p(X["Sperm_Concentration_million_mL"])
        X["MII_Rate"] = X["MII_Oocytes"] / (X["Oocytes_Retrieved"] + 1)
        X["Fertilization_Rate"] = X["Zygotes_2PN"] / (X["MII_Oocytes"] + 1)
        X["Day3_Survival_Rate"] = X["Day3_Embryos_Total"] / (X["Zygotes_2PN"] + 1)
        X["Embryo_Load_Ratio"] = X["Day3_Embryos_Graded"] / (X["Oocytes_Retrieved"] + 1)
        X["Age_AMH_Interaction"] = X["Female_Age"] * X["AMH_ng_mL"]
        X["FSH_LH_Ratio"] = X["Basal_FSH_mIU_mL"] / (X["Basal_LH_mIU_mL"] + 0.01)
        X["Fragmentation_Risk"] = (X["Day3_Fragmentation_%"] > 20).astype(int)
        X["O2_Deviation"] = np.abs(X["O2_Concentration_%"] - 5)
        X["CO2_Deviation"] = np.abs(X["CO2_Concentration_%"] - 6)

        DROP_COLS = [
            "Sperm_Concentration_million_mL",
            "MII_Oocytes",
            "Zygotes_2PN",
            "Day3_Embryos_Total",
            "Day3_Embryos_Graded",
            "Day3_Fragmentation_%",
            "O2_Concentration_%",
            "CO2_Concentration_%"
        ]

        return X.drop(columns=DROP_COLS, errors="ignore")


# Load Model Bundle
@st.cache_resource
def load_bundle():
    bundle = joblib.load("ivf_model_bundle.pkl")
    return bundle

bundle = load_bundle()
model = bundle["pipeline"]
EXPECTED_COLS = bundle["feature_columns"]
THRESHOLD = bundle["threshold"]


# Align Features
def align_features(df):
    return df.reindex(columns=EXPECTED_COLS)


# UI
st.set_page_config(page_title="IVF Blastocyst Prediction", layout="centered")
st.title("IVF Blastocyst Formation Prediction")

tab1, tab2 = st.tabs(["Single Patient", " Bulk Upload"])


# SINGLE PATIENT
with tab1:
    st.subheader("Single IVF Cycle")

    with st.form("single"):
        Female_Age = st.number_input("Female Age", 18, 50, 30)
        BMI = st.number_input("BMI", 15.0, 40.0, 22.0)
        AMH = st.number_input("AMH (ng/mL)", 0.1, 15.0, 2.5)
        AFC = st.number_input("AFC", 0, 40, 10)
        Basal_FSH = st.number_input("Basal FSH", 1.0, 30.0, 6.0)
        Basal_LH = st.number_input("Basal LH", 1.0, 30.0, 5.0)
        Oocytes_Retrieved = st.number_input("Oocytes Retrieved", 0, 40, 10)
        MII_Oocytes = st.number_input("MII Oocytes", 0, 40, 8)
        Zygotes_2PN = st.number_input("Zygotes 2PN", 0, 40, 6)
        Day3_Embryos_Total = st.number_input("Day 3 Embryos", 0, 40, 5)
        Day3_Embryos_Graded = st.number_input("Day 3 Graded", 0, 40, 4)
        Day3_Fragmentation = st.number_input("Day 3 Fragmentation %", 0.0, 100.0, 15.0)
        Sperm_Concentration = st.number_input("Sperm Concentration", 0.1, 200.0, 40.0)
        Insemination_Method = st.selectbox("Insemination Method", ["IVF", "ICSI"])
        Infertility_Diagnosis = st.selectbox(
            "Infertility Diagnosis",
            ["Male Factor", "PCOS", "Endometriosis", "Tubal", "Unexplained", "Diminished Ovarian Reserve"]
        )
        O2 = st.number_input("O2 %", 1.0, 10.0, 5.0)
        CO2 = st.number_input("CO2 %", 1.0, 10.0, 6.0)

        submit = st.form_submit_button("Predict")

    if submit:
        df = pd.DataFrame([{
            "Female_Age": Female_Age,
            "BMI": BMI,
            "AMH_ng_mL": AMH,
            "AFC": AFC,
            "Basal_FSH_mIU_mL": Basal_FSH,
            "Basal_LH_mIU_mL": Basal_LH,
            "Oocytes_Retrieved": Oocytes_Retrieved,
            "MII_Oocytes": MII_Oocytes,
            "Zygotes_2PN": Zygotes_2PN,
            "Day3_Embryos_Total": Day3_Embryos_Total,
            "Day3_Embryos_Graded": Day3_Embryos_Graded,
            "Day3_Fragmentation_%": Day3_Fragmentation,
            "Sperm_Concentration_million_mL": Sperm_Concentration,
            "Insemination_Method": Insemination_Method,
            "Infertility_Diagnosis": Infertility_Diagnosis,
            "O2_Concentration_%": O2,
            "CO2_Concentration_%": CO2
        }])

        df = align_features(df)
        prob = model.predict_proba(df)[0][1]

        st.metric("Blastocyst Probability", f"{prob:.2%}")
        st.success("Likely Formed" if prob >= THRESHOLD else "Not Likely Formed")


# BULK UPLOAD
with tab2:
    st.subheader("Bulk IVF Prediction")
    file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    if file:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        df_aligned = align_features(df)

        probs = model.predict_proba(df_aligned)[:, 1]
        df["Blastocyst_Probability"] = probs
        df["Blastocyst_Prediction"] = np.where(probs >= THRESHOLD, "Likely Formed", "Not Likely Formed")

        st.dataframe(df.head())
        st.download_button("Download Results", df.to_csv(index=False), "results.csv")
