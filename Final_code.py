import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# 1. CONFIGURATION & PAGE SETUP
st.set_page_config(page_title="IVF Blastocyst Prediction", layout="centered")

# 2. FEATURE ENGINEERING CLASS (Must be global)
class IVFDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Ensure numeric safety
        num_cols = X.select_dtypes(exclude="object").columns
        # Handle potential errors if columns are missing during transformation
        existing_num_cols = [c for c in num_cols if c in X.columns]
        X[existing_num_cols] = X[existing_num_cols].apply(pd.to_numeric, errors="coerce")

        # Feature Creation - using .get() to avoid KeyErrors if cols missing
        if "Sperm_Concentration_million_mL" in X.columns:
            X["Sperm_Concentration_log"] = np.log1p(X["Sperm_Concentration_million_mL"])
        
        if all(c in X.columns for c in ["MII_Oocytes", "Oocytes_Retrieved"]):
            X["MII_Rate"] = X["MII_Oocytes"] / (X["Oocytes_Retrieved"] + 1)
            
        if all(c in X.columns for c in ["Zygotes_2PN", "MII_Oocytes"]):
            X["Fertilization_Rate"] = X["Zygotes_2PN"] / (X["MII_Oocytes"] + 1)
            
        if all(c in X.columns for c in ["Day3_Embryos_Total", "Zygotes_2PN"]):
            X["Day3_Survival_Rate"] = X["Day3_Embryos_Total"] / (X["Zygotes_2PN"] + 1)
            
        if all(c in X.columns for c in ["Day3_Embryos_Graded", "Oocytes_Retrieved"]):
            X["Embryo_Load_Ratio"] = X["Day3_Embryos_Graded"] / (X["Oocytes_Retrieved"] + 1)
            
        if all(c in X.columns for c in ["Female_Age", "AMH_ng_mL"]):
            X["Age_AMH_Interaction"] = X["Female_Age"] * X["AMH_ng_mL"]
            
        if all(c in X.columns for c in ["Basal_FSH_mIU_mL", "Basal_LH_mIU_mL"]):
            X["FSH_LH_Ratio"] = X["Basal_FSH_mIU_mL"] / (X["Basal_LH_mIU_mL"] + 0.01)
            
        if "Day3_Fragmentation_%" in X.columns:
            X["Fragmentation_Risk"] = (X["Day3_Fragmentation_%"] > 20).astype(float)
            
        if "O2_Concentration_%" in X.columns:
            X["O2_Deviation"] = np.abs(X["O2_Concentration_%"] - 5)
            
        if "CO2_Concentration_%" in X.columns:
            X["CO2_Deviation"] = np.abs(X["CO2_Concentration_%"] - 6)

        DROP_COLS = [
            "Sperm_Concentration_million_mL", "MII_Oocytes", "Zygotes_2PN",
            "Day3_Embryos_Total", "Day3_Embryos_Graded", "Day3_Fragmentation_%",
            "O2_Concentration_%", "CO2_Concentration_%"
        ]

        return X.drop(columns=DROP_COLS, errors="ignore")


# 3. TRAINING LOGIC (Strictly uses your data)
@st.cache_resource
def train_and_load_model():
    DATA_FILE = "Blastocyst_Formation_Dataset.xlsx"
    
    if not os.path.exists(DATA_FILE):
        return None
    # --- 1. Load Data ---
    try:
        df = pd.read_excel(DATA_FILE, sheet_name=0)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

    # --- 2. Prep Data ---
    TARGET_COL = "Blastocyst_Formation_Flag"
    LEAKAGE_COLS = [
        "Day5_Blastocysts_Formed", "Day5_Blastocysts_Graded", "Expansion_Grade",
        "ICM_Grade", "TE_Grade", "Blastocyst_Quality_Class", "Usable_for_Transfer",
        "Implantation_Success", "Patient_ID", "Cycle_ID", "Lab_ID"
    ]
    
    # Check if target exists
    if TARGET_COL not in df.columns:
        st.error(f"Target column '{TARGET_COL}' not found in dataset.")
        return None

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + LEAKAGE_COLS, errors="ignore")
    
    # --- 3. Define Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), make_column_selector(dtype_exclude=object)),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), make_column_selector(dtype_include=object)),
        ]
    )

    model = GradientBoostingClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=3, 
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )

    pipeline = Pipeline([
        ("feature_engineering", IVFDomainFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # --- 4. Train ---
    pipeline.fit(X, y)
    
    return {
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "threshold": 0.5
    }


# 4. INITIALIZE APP & HANDLE MISSING DATA
st.title("IVF Blastocyst Formation Prediction")

bundle = train_and_load_model()

if bundle is None:
    st.error("❌ **Dataset Not Found**")
    st.warning("Please place the file `Blastocyst_Formation_Dataset.xlsx` in the same folder as this script and refresh the page.")
    st.stop()  # Stop execution here if no model

# Load model components if training succeeded
model = bundle["pipeline"]
EXPECTED_COLS = bundle["feature_columns"]
THRESHOLD = bundle["threshold"]

def align_features(df):
    """Aligns input dataframe columns with training data columns."""
    for col in EXPECTED_COLS:
        if col not in df.columns:
            # Fill missing columns (e.g., missing one-hot categories) with 0
            df[col] = 0
    return df[EXPECTED_COLS]


# 5. UI TABS
tab1, tab2 = st.tabs(["Single Patient Prediction", "Bulk CSV Upload"])

# --- TAB 1: SINGLE PATIENT ---
with tab1:
    st.subheader("Enter IVF Cycle Parameters")
    
    with st.form("single"):
        col1, col2 = st.columns(2)
        
        with col1:
            Female_Age = st.number_input("Female Age", 18, 50, 30)
            BMI = st.number_input("BMI", 15.0, 40.0, 22.0)
            AMH = st.number_input("AMH (ng/mL)", 0.1, 15.0, 2.5)
            AFC = st.number_input("AFC", 0, 40, 10)
            Basal_FSH = st.number_input("Basal FSH", 1.0, 30.0, 6.0)
            Basal_LH = st.number_input("Basal LH", 1.0, 30.0, 5.0)
            Oocytes_Retrieved = st.number_input("Oocytes Retrieved", 0, 40, 10)
            MII_Oocytes = st.number_input("MII Oocytes", 0, 40, 8)
            
        with col2:
            Zygotes_2PN = st.number_input("Zygotes 2PN", 0, 40, 6)
            Day3_Embryos_Total = st.number_input("Day 3 Embryos", 0, 40, 5)
            Day3_Embryos_Graded = st.number_input("Day 3 Graded", 0, 40, 4)
            Day3_Fragmentation = st.number_input("Day 3 Fragmentation %", 0.0, 100.0, 15.0)
            Sperm_Concentration = st.number_input("Sperm Concentration", 0.1, 200.0, 40.0)
            O2 = st.number_input("O2 %", 1.0, 10.0, 5.0)
            CO2 = st.number_input("CO2 %", 1.0, 10.0, 6.0)

        Insemination_Method = st.selectbox("Insemination Method", ["IVF", "ICSI"])
        Infertility_Diagnosis = st.selectbox(
            "Infertility Diagnosis",
            ["Male Factor", "PCOS", "Endometriosis", "Tubal", "Unexplained", "Diminished Ovarian Reserve"]
        )

        submit = st.form_submit_button("Predict Outcome")

    if submit:
        # Create input dataframe
        input_data = {
            "Female_Age": Female_Age, "BMI": BMI, "AMH_ng_mL": AMH, "AFC": AFC,
            "Basal_FSH_mIU_mL": Basal_FSH, "Basal_LH_mIU_mL": Basal_LH,
            "Oocytes_Retrieved": Oocytes_Retrieved, "MII_Oocytes": MII_Oocytes,
            "Zygotes_2PN": Zygotes_2PN, "Day3_Embryos_Total": Day3_Embryos_Total,
            "Day3_Embryos_Graded": Day3_Embryos_Graded, "Day3_Fragmentation_%": Day3_Fragmentation,
            "Sperm_Concentration_million_mL": Sperm_Concentration,
            "Insemination_Method": Insemination_Method,
            "Infertility_Diagnosis": Infertility_Diagnosis,
            "O2_Concentration_%": O2, "CO2_Concentration_%": CO2
        }
        
        df_input = pd.DataFrame([input_data])
        df_aligned = align_features(df_input)
        
        # Predict
        prob = model.predict_proba(df_aligned)[0][1]
        
        st.divider()
        st.metric("Blastocyst Probability", f"{prob:.2%}")
        
        if prob >= THRESHOLD:
            st.success("Likely to Form Blastocyst")
        else:
            st.warning("Not Likely to Form Blastocyst")

# --- TAB 2: BULK UPLOAD ---
with tab2:
    st.subheader("Bulk Prediction (CSV/Excel)")
    file = st.file_uploader("Upload File", type=["csv", "xlsx"])

    if file:
        try:
            df_bulk = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            
            # Align and Predict
            df_bulk_aligned = align
