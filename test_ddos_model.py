import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from sklearn.metrics import classification_report

# =========================
# 1. Load saved preprocessing artifacts
# =========================
with open("encoders/categorical_encoders.pkl", "rb") as f:
    cat_encoders = pickle.load(f)

with open("encoders/label_encoder.pkl", "rb") as f:
    le_label = pickle.load(f)

with open("encoders/categorical_cols.pkl", "rb") as f:
    categorical_cols = pickle.load(f)

with open("encoders/numeric_cols.pkl", "rb") as f:
    numeric_cols = pickle.load(f)

with open("encoders/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================
# 2. Load test dataset
# =========================
test_files = [
    "Datasets/Test Dataset/UDPLag_data_2_0.csv",
    "Datasets/Test Dataset/Syn.csv",
    "Datasets/Test Dataset/DrDoS_NTP_data_data_5.csv",
    "Datasets/Test Dataset/BENIGN.csv"
]

dfs = []
for file in test_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    dfs.append(df)

test_df = pd.concat(dfs, ignore_index=True)

label_col = "Label"

# =========================
# 3. Encode categorical features (handle unseen values safely)
# =========================
for col in categorical_cols:
    if col not in test_df.columns:
        raise ValueError(f"Missing column {col} in test set")
    test_df[col] = test_df[col].astype(str)
    le_cat = cat_encoders[col]
    # Handle unseen labels by mapping them to a default category
    test_df[col] = test_df[col].map(lambda x: x if x in le_cat.classes_ else le_cat.classes_[0])
    test_df[col] = le_cat.transform(test_df[col])

# =========================
# 4. Encode labels
# =========================
if label_col in test_df.columns:
    test_df[label_col] = le_label.transform(test_df[label_col])

# =========================
# 5. Prepare numeric features
# =========================
for col in numeric_cols:
    if col not in test_df.columns:
        raise ValueError(f"Missing numeric column {col} in test set")

test_df[numeric_cols] = test_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())
test_df[numeric_cols] = np.clip(test_df[numeric_cols], -1e9, 1e9)
test_num = scaler.transform(test_df[numeric_cols])

# =========================
# 6. Load model
# =========================
model = tf.keras.models.load_model("ddosAttackDetection.keras")

# =========================
# 7. Prepare inputs for model
# =========================
cat_inputs = [np.array(test_df[col]) for col in categorical_cols]
num_input = np.array(test_num)

# =========================
# 8. Evaluate
# =========================
y_true = test_df[label_col]
y_pred = np.argmax(model.predict(cat_inputs + [num_input]), axis=1)

acc = (y_pred == y_true).mean()
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_true, y_pred, target_names=le_label.classes_, zero_division=0))
