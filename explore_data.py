import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel file
file_path = "data/default of credit card clients.xls"
df = pd.read_excel(file_path, header=1)  # header=1 skips the weird top row

# Basic info
print("📊 Dataset Shape:", df.shape)
print("\n📋 Columns:\n", df.columns.tolist())
print("\n🧹 Missing values:\n", df.isnull().sum())
print("\n📈 Summary Stats:\n", df.describe())

# Rename target column
df.rename(columns={"default payment next month": "default"}, inplace=True)

# Visualize class balance
sns.countplot(data=df, x="default")
plt.title("Class Distribution (0 = Paid, 1 = Defaulted)")
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()
