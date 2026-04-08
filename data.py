import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import tensorflow as tf

import tensorflow as tf

Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Bidirectional = tf.keras.layers.Bidirectional
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
# =========================
# 1. CONFIG
# =========================
TRAIN_PATH = "train.csv"

TEXT_COLUMN = "comment_text"
TARGET_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

MAX_WORDS = 30000
MAX_LEN = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_toxicity_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 2. DOWNLOAD NLTK FILES
# =========================
nltk.download("punkt_tab")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

# =========================
# 3. TEXT PREPROCESSING
# =========================
def clean_text(text: str) -> str:
    """
    Clean text by lowering case, removing URLs, HTML, punctuation,
    numbers, extra spaces, tokenizing, and removing stopwords.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]

    return " ".join(tokens)

# =========================
# 4. LOAD AND EXPLORE DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def explore_data(df: pd.DataFrame) -> None:
    print("\n========== DATA EXPLORATION ==========")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSample rows:\n", df.head())

    print("\nLabel distribution:")
    for col in TARGET_COLUMNS:
        print(f"{col}:")
        print(df[col].value_counts())
        print("-" * 40)

# =========================
# 5. PREPARE DATA
# =========================
def prepare_data(df: pd.DataFrame):
    df = df[[TEXT_COLUMN] + TARGET_COLUMNS].copy()
    df = df.dropna(subset=[TEXT_COLUMN])

    print("\nCleaning text...")
    df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

    X = df["clean_text"]
    y = df[TARGET_COLUMNS].values

    return df, X, y

# =========================
# 6. TOKENIZATION / VECTORIZATION
# =========================
def tokenize_data(X_train, X_val):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    return tokenizer, X_train_pad, X_val_pad

# =========================
# 7. BUILD LSTM MODEL
# =========================
def build_model():
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(TARGET_COLUMNS), activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================
# 8. TRAIN MODEL
# =========================
def train_model(model, X_train_pad, y_train, X_val_pad, y_val):
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history

# =========================
# 9. EVALUATE MODEL
# =========================
def evaluate_model(model, X_val_pad, y_val):
    print("\n========== VALIDATION EVALUATION ==========")

    y_pred_prob = model.predict(X_val_pad)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    subset_acc = np.mean(np.all(y_pred == y_val, axis=1))
    ham_loss = hamming_loss(y_val, y_pred)

    print(f"Subset Accuracy: {subset_acc:.4f}")
    print(f"Hamming Loss: {ham_loss:.4f}")

    for i, label in enumerate(TARGET_COLUMNS):
        print(f"\n--- {label.upper()} ---")
        print(classification_report(y_val[:, i], y_pred[:, i], digits=4))

# =========================
# 10. SAVE TOKENIZER + LABELS + HISTORY
# =========================
def save_artifacts(tokenizer, history):
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(TARGET_COLUMNS, f)

    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print("\nArtifacts saved successfully.")
    print("Model:", MODEL_PATH)
    print("Tokenizer:", TOKENIZER_PATH)
    print("Labels:", LABELS_PATH)
    print("History:", HISTORY_PATH)

# =========================
# 11. MAIN
# =========================
def main():
    df = load_data(TRAIN_PATH)
    explore_data(df)

    df, X, y = prepare_data(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nTrain size:", len(X_train))
    print("Validation size:", len(X_val))

    tokenizer, X_train_pad, X_val_pad = tokenize_data(X_train, X_val)

    model = build_model()
    model.summary()

    history = train_model(model, X_train_pad, y_train, X_val_pad, y_val)
    evaluate_model(model, X_val_pad, y_val)
    save_artifacts(tokenizer, history)

if __name__ == "__main__":
    main()