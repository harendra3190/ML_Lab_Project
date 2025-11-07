# ===============================================================
# CUSTOMER CHURN PREDICTION MODEL - TRAINING PIPELINE
# ===============================================================

# ===================== IMPORTS =====================
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, classification_report,
    roc_curve, auc, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning

# ===================== CONFIG =====================
warnings.filterwarnings('ignore', category=ConvergenceWarning)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("CUSTOMER CHURN PREDICTION MODEL TRAINING")
print("=" * 60)


# ===============================================================
# STEP 1: LOAD DATA
# ===============================================================
print("\n[1/6] Loading dataset...")
df = pd.read_csv('Telco-Customer-Churn.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


# ===============================================================
# STEP 2: DATA PREPROCESSING
# ===============================================================
print("\n[2/6] Data preprocessing...")

# --- Drop unnecessary columns ---
df = df.drop('customerID', axis=1)
print("✓ Dropped customerID column")

# --- Handle missing values ---
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBlank values in TotalCharges: {(df['TotalCharges'] == ' ').sum()}")

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_total = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total)
print(f"✓ Handled TotalCharges missing values (filled with median: {median_total:.2f})")

# --- Separate features and target ---
X = df.drop('Churn', axis=1)
y = df['Churn']

# --- Encode target variable ---
le_target = LabelEncoder()
y = le_target.fit_transform(y)  # No=0, Yes=1

# --- Identify column types ---
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")


# ===============================================================
# FEATURE ENGINEERING
# ===============================================================
print("\n--- Feature Engineering ---")

X_eng = X.copy()

# Derived numerical features
X_eng['AvgMonthlyCharge'] = np.where(
    X_eng['tenure'] > 0, X_eng['TotalCharges'] / X_eng['tenure'], X_eng['MonthlyCharges']
)
X_eng['TenureGroup'] = pd.cut(X_eng['tenure'], bins=[-1, 12, 24, 48, 72],
                              labels=[0, 1, 2, 3]).astype(int)
X_eng['MonthlyChargeGroup'] = pd.cut(X_eng['MonthlyCharges'], bins=[0, 35, 70, 90, 120],
                                     labels=[0, 1, 2, 3]).astype(int)
X_eng['ChargeRatio'] = np.where(X_eng['MonthlyCharges'] > 0,
                                X_eng['TotalCharges'] / X_eng['MonthlyCharges'], 0)

# Service count feature
service_cols = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
X_eng['ServiceCount'] = sum((X_eng[col] == 'Yes').astype(int) for col in service_cols if col in X_eng.columns)

# Encoded numeric mappings
X_eng['InternetService_numeric'] = X_eng['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
X_eng['Contract_numeric'] = X_eng['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
X_eng['PaymentRisk'] = (X_eng['PaymentMethod'] == 'Electronic check').astype(int)

# Interaction features
X_eng['Tenure_x_MonthlyCharge'] = X_eng['tenure'] * X_eng['MonthlyCharges']
X_eng['Contract_x_Tenure'] = X_eng['Contract_numeric'] * X_eng['tenure']

print("✓ Feature engineering completed successfully.")


# ===============================================================
# ENCODING
# ===============================================================
le_dict = {}
X_encoded = X_eng.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    le_dict[col] = le
print("✓ Encoded categorical variables")


# ===============================================================
# STEP 3: TRAIN-TEST SPLIT
# ===============================================================
print("\n[3/6] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Small validation split from training for threshold tuning (no leakage)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
for tr_idx, val_idx in sss.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]


# ===============================================================
# STEP 4: FEATURE SCALING
# ===============================================================
print("\n[4/6] Feature scaling...")
scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled using RobustScaler")


# ===============================================================
# STEP 5: FEATURE SELECTION
# ===============================================================
print("\n[5/7] Feature selection...")
selector = SelectKBest(f_classif, k='all')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"✓ Selected {len(selected_features)} features using SelectKBest")


# ===============================================================
# STEP 6: HANDLE CLASS IMBALANCE
# ===============================================================
print("\n[6/7] Handling class imbalance with SMOTE...")
print(f"Before SMOTE - Stay: {np.sum(y_train==0)}, Churn: {np.sum(y_train==1)}")

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_tr_scaled, y_tr)

print(f"After SMOTE - Stay: {np.sum(y_train_resampled==0)}, Churn: {np.sum(y_train_resampled==1)}")
print("✓ Applied SMOTE for balanced dataset")


# ===============================================================
# STEP 7: MODEL TRAINING & OPTIMIZATION
# ===============================================================
print("\n[7/7] Training models with optimization...")

# --- Logistic Regression ---
print("\n--- Logistic Regression (Optimized) ---")
lr_param_grid = [
    {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1.0, 10.0, 50.0],
     'class_weight': [None, 'balanced', {0:1, 1:3}], 'max_iter':[4000]},
    {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C':[0.01,0.1,1.0,10.0],
     'class_weight':[None, 'balanced'], 'max_iter':[4000]},
]

lr_base = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(lr_base, lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
lr_grid.fit(X_train_resampled, y_train_resampled)
lr_model = lr_grid.best_estimator_

# Threshold tuning on validation
lr_proba_val = lr_model.predict_proba(X_val_scaled)[:, 1]
best_thr_lr, best_acc_lr = 0.5, 0.0
for t in np.linspace(0.3, 0.7, 41):
    acc = accuracy_score(y_val, (lr_proba_val >= t).astype(int))
    if acc > best_acc_lr:
        best_acc_lr = acc
        best_thr_lr = t

lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_pred = (lr_proba >= best_thr_lr).astype(int)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"✓ Best Params: {lr_grid.best_params_}")
print(f"✓ Logistic Regression Accuracy: {lr_accuracy:.4f}")


# --- Decision Tree (Randomized Search) ---
print("\n--- Decision Tree (RandomizedSearch) ---")
dt_param_dist = {
    'max_depth':[8,10,12,14,16], 'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,3,5], 'criterion':['gini','entropy']
}
dt_base = DecisionTreeClassifier(random_state=42)
dt_search = RandomizedSearchCV(dt_base, dt_param_dist, n_iter=50, cv=3, n_jobs=-1, random_state=42)
dt_search.fit(X_tr, y_tr)
dt_model = dt_search.best_estimator_
dt_proba_val = dt_model.predict_proba(X_val)[:,1]
best_thr_dt, best_acc_dt = 0.5, 0.0
for t in np.linspace(0.3, 0.7, 41):
    acc = accuracy_score(y_val, (dt_proba_val >= t).astype(int))
    if acc > best_acc_dt:
        best_acc_dt = acc
        best_thr_dt = t

dt_proba = dt_model.predict_proba(X_test)[:,1]
dt_pred = (dt_proba >= best_thr_dt).astype(int)

# DT metrics
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_auc = auc(*roc_curve(y_test, dt_proba)[:2])
print(f"✓ Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"✓ Decision Tree Precision: {dt_precision:.4f}  Recall: {dt_recall:.4f}  F1: {dt_f1:.4f}")
print(f"✓ Decision Tree ROC AUC: {dt_auc:.4f}")


# --- Random Forest ---
print("\n--- Random Forest (RandomizedSearch) ---")
rf_param_grid = {
    'n_estimators':[200,300,400],
    'max_depth':[16,20,24,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,3,5],
    'max_features':['sqrt','log2'],
    'class_weight':[None,'balanced']
}
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = RandomizedSearchCV(rf_base, rf_param_grid, cv=5, n_iter=20, n_jobs=-1, random_state=42)
rf_grid.fit(X_tr, y_tr)
rf_model = rf_grid.best_estimator_
rf_proba_val = rf_model.predict_proba(X_val)[:,1]
best_thr_rf, best_acc_rf = 0.5, 0.0
for t in np.linspace(0.3, 0.7, 41):
    acc = accuracy_score(y_val, (rf_proba_val >= t).astype(int))
    if acc > best_acc_rf:
        best_acc_rf = acc
        best_thr_rf = t

rf_proba = rf_model.predict_proba(X_test)[:,1]
rf_pred = (rf_proba >= best_thr_rf).astype(int)

# RF metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_auc = auc(*roc_curve(y_test, rf_proba)[:2])
print(f"✓ Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"✓ Random Forest Precision: {rf_precision:.4f}  Recall: {rf_recall:.4f}  F1: {rf_f1:.4f}")
print(f"✓ Random Forest ROC AUC: {rf_auc:.4f}")


# --- Voting Classifier ---
print("\n--- Voting Classifier (Ensemble) ---")
voting_clf = VotingClassifier(
    estimators=[('lr', lr_model), ('dt', dt_model), ('rf', rf_model)],
    voting='soft'
)
voting_clf.fit(X_train_resampled, y_train_resampled)
v_proba_val = voting_clf.predict_proba(X_val_scaled)[:,1]
best_thr_v, best_acc_v = 0.5, 0.0
for t in np.linspace(0.3, 0.7, 41):
    acc = accuracy_score(y_val, (v_proba_val >= t).astype(int))
    if acc > best_acc_v:
        best_acc_v = acc
        best_thr_v = t

voting_proba = voting_clf.predict_proba(X_test_scaled)[:,1]
voting_pred = (voting_proba >= best_thr_v).astype(int)

# Voting metrics
v_accuracy = accuracy_score(y_test, voting_pred)
v_f1 = f1_score(y_test, voting_pred)
v_precision = precision_score(y_test, voting_pred)
v_recall = recall_score(y_test, voting_pred)
v_auc = auc(*roc_curve(y_test, voting_proba)[:2])
print(f"✓ Voting Accuracy: {v_accuracy:.4f}")
print(f"✓ Voting Precision: {v_precision:.4f}  Recall: {v_recall:.4f}  F1: {v_f1:.4f}")
print(f"✓ Voting ROC AUC: {v_auc:.4f}")


# ===============================================================
# EVALUATION, VISUALIZATION & SAVING MODELS
# ===============================================================
print("\n[8/8] Evaluation & Saving...")

# --- Feature Importance ---
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop Features:")
print(feature_importance.head(5))

# --- Save Models ---
joblib.dump(lr_model, 'model data/ls/lr_model.pkl')
joblib.dump(dt_model, 'model data/ls/dt_model.pkl')
joblib.dump(rf_model, 'model data/ls/rf_model.pkl')
joblib.dump(voting_clf, 'model data/ls/voting_model.pkl')
joblib.dump(le_dict, 'model data/ls/label_encoders.pkl')
joblib.dump(le_target, 'model data/ls/target_encoder.pkl')
joblib.dump(list(X_train.columns), 'model data/ls/feature_names.pkl')
print("\n✓ All models and encoders saved successfully.")

print("\n" + "="*60)
print("🎯 TRAINING COMPLETE")
print("="*60)
