import re
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences

nltk.download("punkt_tab")
nltk.download("stopwords")

# =========================
# CONFIG
# =========================
MODEL_PATH = "artifacts/lstm_toxicity_model.keras"
TOKENIZER_PATH = "artifacts/tokenizer.pkl"
LABELS_PATH = "artifacts/labels.pkl"
HISTORY_PATH = "artifacts/training_history.pkl"

MAX_LEN = 200
STOP_WORDS = set(stopwords.words("english"))

# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_labels():
    with open(LABELS_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_history():
    with open(HISTORY_PATH, "rb") as f:
        return pickle.load(f)

# =========================
# TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
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

def predict_text(text, model, tokenizer, labels):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    pred = model.predict(padded)[0]

    result = {label: float(score) for label, score in zip(labels, pred)}
    return result

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Comment Toxicity Detection", layout="wide")
st.title("Comment Toxicity Detection with LSTM")
st.write("Predict toxicity labels for single comments or bulk CSV uploads.")

model = load_model()
tokenizer = load_tokenizer()
labels = load_labels()
history = load_history()

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Prediction",
    "Bulk Prediction",
    "Model Metrics",
    "Sample Test Cases"
])

# =========================
# TAB 1: SINGLE PREDICTION
# =========================
with tab1:
    st.subheader("Single Comment Prediction")

    user_input = st.text_area("Enter a comment", height=150)

    if st.button("Predict"):
        if user_input.strip():
            result = predict_text(user_input, model, tokenizer, labels)

            st.write("### Prediction Scores")
            result_df = pd.DataFrame({
                "Label": list(result.keys()),
                "Score": list(result.values())
            })
            st.dataframe(result_df, use_container_width=True)

            toxic_any = any(score >= 0.5 for score in result.values())
            st.write("### Final Status")
            if toxic_any:
                st.success("Toxic comment detected")
            else:
                st.info("Non-toxic comment")

# =========================
# TAB 2: BULK PREDICTION
# =========================
with tab2:
    st.subheader("Bulk CSV Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        if "comment_text" not in df.columns:
            st.error("CSV must contain a 'comment_text' column.")
        else:
            if st.button("Run Bulk Prediction"):
                predictions = []

                for text in df["comment_text"].fillna(""):
                    result = predict_text(text, model, tokenizer, labels)
                    predictions.append(result)

                pred_df = pd.DataFrame(predictions)
                output_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

                st.write("### Prediction Output")
                st.dataframe(output_df.head(), use_container_width=True)

                csv = output_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions CSV",
                    data=csv,
                    file_name="bulk_toxicity_predictions.csv",
                    mime="text/csv"
                )

# =========================
# TAB 3: MODEL METRICS
# =========================
with tab3:
    st.subheader("Training Metrics")

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)

        if "loss" in history_df.columns:
            st.line_chart(history_df[["loss", "val_loss"]])

        if "accuracy" in history_df.columns:
            st.line_chart(history_df[["accuracy", "val_accuracy"]])
    else:
        st.info("Training history not found.")

# =========================
# TAB 4: SAMPLE TEST CASES
# =========================
with tab4:
    st.subheader("Sample Test Cases")

    samples = [
        "Thank you for your help, I really appreciate it.",
        "You are so stupid and disgusting.",
        "I will find you and hurt you.",
        "This is a normal discussion comment.",
        "Shut up, you idiot."
    ]

    for sample in samples:
        st.write(f"**Comment:** {sample}")
        result = predict_text(sample, model, tokenizer, labels)
        st.write(result)
        st.markdown("---")