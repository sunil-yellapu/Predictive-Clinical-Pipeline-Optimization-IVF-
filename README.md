#  Data-Driven Blastocyst Formation Prediction

## Project Overview

This project focuses on predicting **blastocyst formation** using a **data-driven, clinically sensible machine learning approach**. Rather than blindly experimenting with multiple algorithms, the modeling strategy is intentionally **phase-based**, emphasizing understanding, reliability, and biological interpretability.

The objective is to arrive at **one robust, well-validated model** that balances performance with trustworthiness—especially critical in a clinical/biological setting.


##  Modeling Philosophy

The core philosophy of this project is **intentional modeling**.

Instead of optimizing everything at once, the approach prioritizes:

* Understanding feature relevance before model complexity
* Comparing model families systematically
* Optimizing only the things that improves outcomes
* Avoiding unnecessary sophistication that reduces interpretability

> **Guiding Principle:** Modeling progresses from *certainty -> comparison -> optimization -> trust*.
---

##  Modeling Workflow (Phase-Based)

### Phase 1 — Baseline Sanity Check

**Objective:** Validate data, preprocessing, and pipeline correctness.

**Approach:**

* Train a simple baseline model (Logistic Regression or Simple Decision Tree)
* No hyperparameter tuning
* Single primary metric (Recall / F1-score)

**Why this phase matters:**

* Confirms preprocessing correctness
* Detects data leakage or logical bugs early
* Establishes a reference performance

> If the baseline fails, the pipeline is debugged before moving forward.

---

### Phase 2 — Model Family Comparison

**Objective:** Identify which class of models is best suited to the data.

**Models evaluated:**

* Random Forest
* Gradient Boosting
* XGBoost / LightGBM
* CatBoost(Optional)

**Rules enforced:**

* Identical preprocessing
* Same train–validation split or cross-validation strategy
* Same evaluation metric
* No aggressive hyperparameter tuning

**Outcome:**

* Shortlist the top 1–2 performing model families

---

### Phase 3 — Deep Optimization (Only the  Shortlisted models)

**Objective:** Improve performance without overfitting.

**Techniques used:**

* RandomizedSearchCV
* Proper handling of class imbalance
* Regularization and depth control

**Evaluation focus:**

* Train vs validation performance gap
* ROC-AUC
* Precision–Recall trade-off

---

### Phase 4 — Reliability & Clinical Sense Check

**Objective:** Ensure the model is reliable and biologically reasonable.

**Checks performed:**

* Cross-validation stability
* Overfitting assessment (Train F1 vs Test F1)
* Threshold tuning (default 0.5 vs domain-relevant thresholds)
* Confusion matrix interpretation

**Key emphasis:**

* Recall and false negatives are prioritized over raw accuracy
* Model behavior must align with biological intuition

---

### Phase 5 — Final Model Selection and Freezing

**Objective:** Lock a single, production-ready model.

**Frozen components:**

* Feature engineering logic
* Preprocessing steps
* Final model and parameters

**Deliverables:**

* End-to-end pipeline
* Final metrics
* Documented assumptions and limitations

---

##  What This Project Intentionally Avoids

To maintain clarity and robustness, the following practices are deliberately excluded:

* Dropping features purely based on correlation
* Aggressive outlier treatment for tree-based models
* Hyperparameter tuning across all models
* Accuracy-first optimization
* Mixing preprocessing logic
