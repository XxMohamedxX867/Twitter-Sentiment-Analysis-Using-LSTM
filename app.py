import streamlit as st
import tensorflow as tf
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import sys

# Fix pickle loading for keras/tensorflow mismatch
sys.modules['keras.preprocessing.text'] = sys.modules['tensorflow.keras.preprocessing.text']

# Load model & tokenizer
model = tf.keras.models.load_model("Sentiment_Analysis_85_test_Acc.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Config
max_len = 100
labels_map = {
    0: ("Negative", "😠"),
    1: ("Positive", "😊"),
    2: ("Neutral", "😐"),
    3: ("Irrelevant", "❓")
}

# NLTK setup
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 3]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")
st.title("🐦 Twitter Sentiment Analysis")
st.write("Enter a tweet and click Predict to see the sentiment.")

user_input = st.text_area("✏️ Type tweet here:", "", placeholder="Example: I love the new iPhone!")

if st.button("Predict"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len)
        pred = model.predict(padded, verbose=0)[0]

        pred_class = np.argmax(pred)
        sentiment, emoji = labels_map[pred_class]

        st.markdown(f"### {emoji} {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")
