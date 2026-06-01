import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text Cleaning Function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Streamlit UI
st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the model will predict whether it is Positive or Negative."
)

review = st.text_area(
    "Enter Movie Review",
    height=150
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        cleaned_review = clean_text(review)

        review_vector = vectorizer.transform([cleaned_review])

        prediction = model.predict(review_vector)

        probability = model.predict_proba(review_vector)

        confidence = round(max(probability[0]) * 100, 2)

        if prediction[0] == 1:
            st.success(
                f"😊 Positive Review\n\nConfidence: {confidence}%"
            )
        else:
            st.error(
                f"😞 Negative Review\n\nConfidence: {confidence}%"
            )

st.markdown("---")
st.caption("Built using NLP, TF-IDF and Logistic Regression")