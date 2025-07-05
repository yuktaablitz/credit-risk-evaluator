import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load processed data
X_train, X_test, y_train, y_test = joblib.load("data/processed_data.pkl")

# --- Model 1: Logistic Regression
print("\n🔍 Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("📊 Logistic Regression:\n", classification_report(y_test, y_pred_log))
print("🎯 ROC AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]))

# --- Model 2: Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("📊 Random Forest:\n", classification_report(y_test, y_pred_rf))
print("🎯 ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# --- Model 3: XGBoost
print("\n⚡ Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("📊 XGBoost:\n", classification_report(y_test, y_pred_xgb))
print("🎯 ROC AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))

# Save best model (e.g. XGBoost)
joblib.dump(xgb, "model/credit_risk_model.pkl")
print("\n✅ Best model (XGBoost) saved to 'model/credit_risk_model.pkl'")
