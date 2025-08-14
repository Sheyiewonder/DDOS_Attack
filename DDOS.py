import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Embedding, Flatten, Concatenate

# =========================
# 1. Load and merge datasets
# =========================
files = [
    "Datasets/Training Dataset/UDPLag_data_2_0.csv",
    "Datasets/Training Dataset/Syn.csv",
    "Datasets/Training Dataset/DrDoS_NTP_data_data_5.csv",
    "Datasets/Training Dataset/BENIGN.csv"
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

# Stratification check before split
print("\nClass distribution in full dataset:")
print(data[label_col].value_counts(normalize=True))

# Split features/labels
X = data[categorical_cols + numeric_cols]
y = data[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nClass distribution in TRAIN:")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nClass distribution in TEST:")
print(pd.Series(y_test).value_counts(normalize=True))

# =========================
# 2.1 Handle inf / NaN in numeric columns
# =========================
for df_part in [X_train, X_test]:
    df_part[numeric_cols] = df_part[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df_part[numeric_cols] = df_part[numeric_cols].fillna(df_part[numeric_cols].median())
    df_part[numeric_cols] = np.clip(df_part[numeric_cols], -1e9, 1e9)  # prevent extreme values

# =========================
# 3. Scale numeric features
# =========================
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numeric_cols])
X_test_num = scaler.transform(X_test[numeric_cols])

# =========================
# 4. Build TabTransformer-style model
# =========================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Dense(ff_dim, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_tabtransformer(cat_dims, num_num_features, num_classes):
    cat_inputs = []
    cat_embeds = []

    # Embeddings for categorical features
    for dim in cat_dims:
        inp = Input(shape=(1,))
        emb = Embedding(input_dim=dim, output_dim=min(50, (dim + 1) // 2))(inp)
        emb = Flatten()(emb)
        cat_inputs.append(inp)
        cat_embeds.append(emb)

    # Numerical features input
    num_input = Input(shape=(num_num_features,))

    # Combine categorical embeddings + numerical
    all_features = Concatenate()(cat_embeds + [num_input])

    # Reshape for transformer (features as tokens)
    x = layers.Reshape((all_features.shape[1], 1))(all_features)

    # Transformer blocks
    x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64, dropout=0.1)
    x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64, dropout=0.1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=cat_inputs + [num_input], outputs=outputs)

# Prepare model inputs
cat_dims = [int(X[col].nunique()) + 1 for col in categorical_cols]
num_classes = len(np.unique(y))

model = build_tabtransformer(cat_dims, len(numeric_cols), num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Convert data for model input
train_cat = [np.array(X_train[col]) for col in categorical_cols]
test_cat = [np.array(X_test[col]) for col in categorical_cols]
train_num = np.array(X_train_num)
test_num = np.array(X_test_num)

# =========================
# 5. Train
# =========================
history = model.fit(
    train_cat + [train_num], y_train,
    validation_data=(test_cat + [test_num], y_test),
    epochs=100,
    batch_size=32
)

# =========================
# 6. Evaluate
# =========================
model.save("ddosAttackDetection.h5")

# =========================
# 6. Evaluate
# =========================
loss, acc = model.evaluate(test_cat + [test_num], y_test)
print(f"Test Accuracy: {acc:.4f}")

y_pred = np.argmax(model.predict(test_cat + [test_num]), axis=1)
print(classification_report(y_test, y_pred, target_names=le_label.classes_, zero_division=0))
