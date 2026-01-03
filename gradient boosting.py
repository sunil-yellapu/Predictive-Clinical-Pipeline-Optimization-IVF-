
# DATA-DRIVEN BLASTOCYST FORMATION PREDICTION

# TRAIN + SAVE FINAL MODEL (STREAMLIT SAFE)

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


# LOAD DATA 
from sqlalchemy import create_engine

user = "root"
pw = "XXXXXXXX"
db = "ivf_db" 

engine = create_engine(
    f"mysql+pymysql://{user}:{pw}@localhost/{db}"
)

df = pd.read_excel( "Blastocyst_Formation_Dataset.xlsx",sheet_name=0)

df.to_sql(
    name="blastocyst_data",
    con=engine,
    if_exists="replace",
    index=False,
    chunksize=1000
)

sql = 'select * from blastocyst_data'
data = pd.read_sql_query(sql, engine)
print(type(data))
df.head(5)
df.describe()
df.info()

df = df.copy()

# Target & Leakage

TARGET_COL = "Blastocyst_Formation_Flag"

LEAKAGE_COLS = [
    "Day5_Blastocysts_Formed",
    "Day5_Blastocysts_Graded",
    "Expansion_Grade",
    "ICM_Grade",
    "TE_Grade",
    "Blastocyst_Quality_Class",
    "Usable_for_Transfer",
    "Implantation_Success",
    "Patient_ID",
    "Cycle_ID",
    "Lab_ID"
]

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL] + LEAKAGE_COLS, errors="ignore")


# Feature Engineering (SAFE)

class IVFDomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ensure numeric safety
        num_cols = X.select_dtypes(exclude="object").columns
        X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")

        X["Sperm_Concentration_log"] = np.log1p(X["Sperm_Concentration_million_mL"])
        X["MII_Rate"] = X["MII_Oocytes"] / (X["Oocytes_Retrieved"] + 1)
        X["Fertilization_Rate"] = X["Zygotes_2PN"] / (X["MII_Oocytes"] + 1)
        X["Day3_Survival_Rate"] = X["Day3_Embryos_Total"] / (X["Zygotes_2PN"] + 1)
        X["Embryo_Load_Ratio"] = X["Day3_Embryos_Graded"] / (X["Oocytes_Retrieved"] + 1)
        X["Age_AMH_Interaction"] = X["Female_Age"] * X["AMH_ng_mL"]
        X["FSH_LH_Ratio"] = X["Basal_FSH_mIU_mL"] / (X["Basal_LH_mIU_mL"] + 0.01)
        X["Fragmentation_Risk"] = (X["Day3_Fragmentation_%"] > 20).astype(float)
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


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Preprocessing (STREAMLIT SAFE)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            make_column_selector(dtype_exclude=object),
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]),
            make_column_selector(dtype_include=object),
        ),
    ]
)


# Model
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42
)

FINAL_THRESHOLD = 0.50


# Pipeline
pipeline = Pipeline([
    ("feature_engineering", IVFDomainFeatureEngineer()),
    ("preprocessing", preprocessor),
    ("model", model)
])


# Train
pipeline.fit(X_train, y_train)


# Evaluation (Optional)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Save EVERYTHING for inference
joblib.dump(
    {
        "pipeline": pipeline,
        "feature_columns": X_train.columns.tolist(),
        "threshold": FINAL_THRESHOLD,
        "model_name": "GradientBoostingClassifier"
    },
    "ivf_model_bundle.pkl"
)

print("Model bundle saved successfully")
