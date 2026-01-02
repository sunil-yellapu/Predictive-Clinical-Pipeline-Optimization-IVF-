# Data-Driven-Blastocyst-Formation-Prediction

🧠 Modeling Strategy (How I Approach This Project)

This section documents how I should think and proceed with modeling, so I don’t get confused or over-engineer later.

## 1️⃣ Goal of Modeling

The goal is not to try every model blindly.

The goal is to:

Build trust in the data

Identify the best model family

Carefully optimize only what matters

End with one reliable model

## 2️⃣ Modeling Is Done in Phases (Not All at Once)
### Phase 1 — Baseline Sanity Check

Train one simple model (e.g., shallow Decision Tree or Logistic Regression)

No hyperparameter tuning

One main metric (F1 / Recall)

Purpose:

Verify preprocessing is correct

Detect leakage or bugs early

Set a reference performance

If this fails → stop and debug.

### Phase 2 — Model Family Comparison

Train a small set of representative models:

  Random Forest
  
  Gradient Boosting
  
  XGBoost / LightGBM
  
  (Optional) CatBoost

Same preprocessing

Same split / CV

Same metric

Rules:

No heavy tuning yet

No threshold tuning

Just compare which model family works best

Outcome:

Pick top 1–2 models only

### Phase 3 — Deep Optimization (Only for Top Models)

Apply RandomizedSearchCV

Handle class imbalance properly

Control overfitting (depth, regularization)

Evaluate:

  Train vs validation gap
  
  ROC-AUC
  
  Precision–Recall tradeoff
  
Purpose:

Improve performance without overfitting

### Phase 4 — Reliability & Clinical Sense Check

Cross-validation on training data

Overfitting check (Train F1 vs Test F1)

Threshold tuning (0.5 vs domain-relevant threshold)

Confusion matrix interpretation

Focus:

Recall / false negatives matter more than accuracy

Model behavior must make biological sense

### Phase 5 — Final Model Freeze

Select one final model

Freeze:

  Feature engineering
  
  Preprocessing
  
  Model parameters

Save a single pipeline

Document final metrics and limitations

## 3️⃣ What I Intentionally Do NOT Do


I do NOT drop features based on correlation

I do NOT aggressively treat outliers (tree models)

I do NOT tune all models

I do NOT optimize accuracy blindly

I do NOT mix preprocessing and modeling logic

## 4️⃣ Key Principle to Remember

Modeling moves from certainty → comparison → optimization → trust.

If I ever feel lost:

  Go back to the phase structure

  Do one phase at a time
  
  Keep decisions intentional

## 5️⃣ Current Status

Data preprocessing: ✅ done
Feature engineering: ✅ done
Correlation analysis: ✅ done

Next step: Baseline model training

This README section is your mental anchor.
Whenever you return to this project, read this first — it will keep you aligned.
