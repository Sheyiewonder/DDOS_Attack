import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# =========================
# 1. Load and merge test datasets
# =========================
files = [
    "Datasets/Test Dataset/UDPLag_data_2_0.csv",
    "Datasets/Test Dataset/Syn.csv",
    "Datasets/Test Dataset/DrDoS_NTP_data_data_5.csv",
    "Datasets/Test Dataset/BENIGN.csv"
]

dfs = []
for file in files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    if "Label" not in df.columns:
        raise ValueError(f"'Label' column missing in {file}. Found: {df.columns.tolist()}")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

label_col = "Label"

# =========================
# 2. Separate categorical & numeric features
# =========================
categorical_cols = data.select_dtypes(exclude=["number"]).columns.tolist()
categorical_cols.remove(label_col) if label_col in categorical_cols else None
numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

# Encode categorical features
for col in categorical_cols:
    le_cat = LabelEncoder()
    data[col] = le_cat.fit_transform(data[col].astype(str))

# Encode labels
le_label = LabelEncoder()
data[label_col] = le_label.fit_transform(data[label_col])

# =========================
# 2.1 Handle inf / NaN in numeric columns
# =========================
data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
data[numeric_cols] = np.clip(data[numeric_cols], -1e9, 1e9)

# =========================
# 3. Scale numeric features
# =========================
scaler = StandardScaler()
X_num = scaler.fit_transform(data[numeric_cols])

# =========================
# 4. Prepare model inputs
# =========================
X_cat = [np.array(data[col]) for col in categorical_cols]
X_num = np.array(X_num)
y_true = data[label_col]

# =========================
# 5. Load model and evaluate
# =========================
model = load_model("ddosAttackDetection.h5")

loss, acc = model.evaluate(X_cat + [X_num], y_true)
print(f"Test Accuracy: {acc:.4f}")

y_pred = np.argmax(model.predict(X_cat + [X_num]), axis=1)
print(classification_report(y_true, y_pred, target_names=le_label.classes_, zero_division=0))