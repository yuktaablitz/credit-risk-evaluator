import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Load the data
df = pd.read_excel("data/default of credit card clients.xls", header=1)

# Rename target column
df.rename(columns={"default payment next month": "default"}, inplace=True)

# Drop the ID column
df.drop(columns=["ID"], inplace=True)

# Separate features and target
X = df.drop("default", axis=1)
y = df["default"]

# Optional: check if categorical columns need encoding
# SEX, EDUCATION, MARRIAGE are already numeric
# But we can still ensure they’re in the correct range

print("\n🔍 Unique values in categorical features:")
for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    print(f"{col}: {sorted(X[col].unique())}")

# Fix EDUCATION and MARRIAGE values if needed
X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # group unknowns as 'others'
X["MARRIAGE"] = X["MARRIAGE"].replace({0: 3})  # unknown marriage as 'other'

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
import joblib
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), "data/processed_data.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Data preparation complete.")
print("📦 Processed data saved to 'data/processed_data.pkl'")
