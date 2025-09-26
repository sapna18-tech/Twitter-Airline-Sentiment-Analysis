#!/usr/bin/env python3
"""
twitter_airline_sentiment.py
Complete end-to-end script for Twitter Airline Sentiment Analysis.

Usage:
    python twitter_airline_sentiment.py
    (See options below - none required for default behavior.)

Outputs:
    - airline_sentiment_logreg_tfidf_online_data.pkl  (trained baseline model)
    - tokenizer.json (if LSTM option used)
    - prints EDA and evaluation metrics to console
"""

import os
import re
import sys
import zipfile
import argparse
import urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# sklearn / preprocessing / models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# simple plotting (if running interactively)
import matplotlib.pyplot as plt

# NLTK (text preprocessing)
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Optional libraries for advanced features
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# For saving
import joblib

# -------------------------
# Config / constants
# -------------------------
GITHUB_RAW_URL = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
LOCAL_FILENAMES = ["Tweets.csv", "data/Tweets.csv", "data/tweets.csv"]

STOPWORDS = set(stopwords.words('english'))

# -------------------------
# Helper functions
# -------------------------
def download_from_github_raw(save_path="Tweets.csv"):
    try:
        print(f"[INFO] Trying to download dataset from GitHub raw: {GITHUB_RAW_URL}")
        urllib.request.urlretrieve(GITHUB_RAW_URL, save_path)
        print("[INFO] Download successful.")
        return Path(save_path)
    except Exception as e:
        print("[WARN] GitHub download failed:", e)
        return None

def download_from_huggingface(save_path="Tweets_hf.csv"):
    if not HF_AVAILABLE:
        print("[INFO] HuggingFace datasets not available; skipping.")
        return None
    try:
        print("[INFO] Loading dataset from HuggingFace 'osanseviero/twitter-airline-sentiment' ...")
        ds = load_dataset("osanseviero/twitter-airline-sentiment")
        df = pd.DataFrame(ds['train'])
        df.to_csv(save_path, index=False)
        print("[INFO] Saved HuggingFace dataset to", save_path)
        return Path(save_path)
    except Exception as e:
        print("[WARN] HuggingFace load failed:", e)
        return None

def try_local_files():
    for f in LOCAL_FILENAMES:
        if Path(f).exists():
            print("[INFO] Found local file:", f)
            return Path(f)
    return None

def fetch_dataset():
    # 1) Try GitHub raw
    p = download_from_github_raw("Tweets.csv")
    if p and p.exists():
        return pd.read_csv(p)

    # 2) Try HuggingFace
    p = download_from_huggingface("Tweets_hf.csv")
    if p and p.exists():
        return pd.read_csv(p)

    # 3) Try local
    p = try_local_files()
    if p:
        return pd.read_csv(p)

    raise FileNotFoundError(
        "Could not fetch dataset automatically. Place 'Tweets.csv' in the working directory "
        "or install 'datasets' library (pip install datasets) for HuggingFace fallback."
    )

def simple_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)             # remove urls
    text = re.sub(r'@\w+', '', text)                # remove mentions
    text = re.sub(r'#', '', text)                   # remove hashtag sign only
    text = re.sub(r'[^a-z0-9\s]', ' ', text)        # keep alphanum
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def top_n_words(series, n=20):
    all_words = " ".join(series).split()
    return Counter(all_words).most_common(n)

# -------------------------
# Main pipeline
# -------------------------
def main(args):
    print("=== Twitter Airline Sentiment Analysis — START ===")

    # 1) Load dataset (try online sources)
    print("\n[STEP] Fetching dataset...")
    df = fetch_dataset()
    print("[INFO] Dataset loaded. Shape:", df.shape)
    if 'airline_sentiment' not in df.columns or 'text' not in df.columns:
        print("[ERROR] Expected columns 'airline_sentiment' and 'text' not found. Columns:", df.columns.tolist())
        return

    # Basic EDA
    print("\n[STEP] Quick EDA")
    print(df['airline_sentiment'].value_counts())
    print("\nSample rows:")
    print(df[['airline_sentiment','airline','text']].head(5).to_string(index=False))

    # 2) Preprocessing
    print("\n[STEP] Preprocessing text (this may take a minute)...")
    tqdm.pandas()
    df['proc_text'] = df['text'].progress_apply(simple_clean)
    # drop rows with empty processed text
    df = df[df['proc_text'].str.strip().astype(bool)].reset_index(drop=True)
    print("[INFO] After cleaning, dataset shape:", df.shape)

    # Show top words
    print("\nTop words in negative tweets:")
    print(top_n_words(df[df['airline_sentiment']=='negative']['proc_text'], 15))
    print("\nTop words in positive tweets:")
    print(top_n_words(df[df['airline_sentiment']=='positive']['proc_text'], 15))

    # Map labels: negative=0, neutral=1, positive=2
    label_map = {'negative':0, 'neutral':1, 'positive':2}
    df['label'] = df['airline_sentiment'].map(label_map)

    # 3) Train/test split
    X = df['proc_text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"\n[INFO] Train/test split: {len(X_train)} train, {len(X_test)} test")

    # 4) Baseline model: TF-IDF + Logistic Regression
    print("\n[STEP] Training baseline: TF-IDF + LogisticRegression ...")
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    clf = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial')
    pipe = Pipeline([('tfidf', tfidf), ('clf', clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print("\n[RESULT] Baseline evaluation:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 (macro):", f1_score(y_test, preds, average='macro'))
    print(classification_report(y_test, preds, target_names=['negative','neutral','positive']))

    # Save model
    model_path = "airline_sentiment_logreg_tfidf_online_data.pkl"
    joblib.dump(pipe, model_path)
    print("[INFO] Saved baseline model to", model_path)

    # 5) Optional: quick classical models comparison
    if args.quick_compare:
        print("\n[STEP] Quick comparison: Logistic / LinearSVC / MultinomialNB")
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial'),
            "LinearSVC": LinearSVC(max_iter=5000),
            "MultinomialNB": MultinomialNB()
        }
        for name, model in models.items():
            p = Pipeline([('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1,2))),
                          ('clf', model)])
            p.fit(X_train, y_train)
            pr = p.predict(X_test)
            print(f"\n=== {name} ===")
            print("Accuracy:", accuracy_score(y_test, pr))
            print("F1 (macro):", f1_score(y_test, pr, average='macro'))
            print(classification_report(y_test, pr, target_names=['negative','neutral','positive']))

    # 6) Optional: quick confusion matrix plot (if matplotlib available & interactive)
    if args.plot_cm:
        print("\n[STEP] Plotting confusion matrix (baseline)...")
        cm = confusion_matrix(y_test, preds, labels=[0,1,2])
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        ax.set_xticklabels(['negative','neutral','positive'], rotation=45)
        ax.set_yticklabels(['negative','neutral','positive'])
        ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title("Confusion Matrix (Baseline)")
        plt.tight_layout()
        plt.show()

    # 7) Optional: LSTM (commented by default — set --use_lstm to run; may require more time & resources)
    if args.use_lstm:
        print("\n[STEP] Building & training a small LSTM model (this may take time)...")
        # Import TensorFlow lazily to avoid requiring it when not requested
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras import layers, models, callbacks
        except Exception as e:
            print("[ERROR] TensorFlow is required for LSTM but not found:", e)
            return

        MAX_WORDS = 20000
        MAX_LEN = 60
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=3)

        embedding_dim = 100
        model = models.Sequential([
            layers.Embedding(input_dim=MAX_WORDS, output_dim=embedding_dim, input_length=MAX_LEN),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.GlobalMaxPool1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(
            X_train_pad, y_train_cat,
            validation_split=0.1,
            epochs=8,
            batch_size=128,
            callbacks=[es],
            verbose=1
        )
        loss, acc = model.evaluate(X_test_pad, y_test_cat, verbose=1)
        print("[LSTM] Test acc:", acc)

        # Save tokenizer and model
        tokenizer_json = tokenizer.to_json()
        with open("tokenizer.json","w") as f:
            f.write(tokenizer_json)
        model.save("lstm_airline_sentiment.h5")
        print("[INFO] Saved LSTM tokenizer + model.")

    print("\n=== Done. Baseline model saved. ===")

# -------------------------
# CLI / args
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twitter Airline Sentiment Analysis — end-to-end script")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set proportion (default 0.2)")
    parser.add_argument("--quick-compare", action='store_true', help="Run quick classical models comparison (SVC, NB)")
    parser.add_argument("--plot-cm", action='store_true', help="Plot confusion matrix (requires matplotlib and interactive environment)")
    parser.add_argument("--use-lstm", action='store_true', help="Train a small LSTM model (requires TensorFlow)")
    args = parser.parse_args()
    main(args)
