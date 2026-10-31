
import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_model():
    # Download required text-processing resources
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    model = joblib.load("naive_bayes_model.pkl")
    vectorizer = joblib.load("count_vectorizer.pkl")

    stop_words = set(stopwords.words("english"))
    stop_words -= {"not", "no", "nor"}

    lemmatizer = WordNetLemmatizer()

    return model, vectorizer, stop_words, lemmatizer


model, vectorizer, stop_words, lemmatizer = load_model()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


st.title("Review Sentiment Analyzer")
st.write(
    "Enter a review to predict whether the sentiment "
    "is positive or negative."
)

review = st.text_area(
    "Write your review",
    placeholder="Example: The food was fantastic!"
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned_review = clean_text(review)
        features = vectorizer.transform([cleaned_review])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("Predicted Sentiment: Positive 😊")
        elif prediction == 0:
            st.error("Predicted Sentiment: Negative")
        else:
            st.info(f"Predicted class: {prediction}")

        with st.expander("View cleaned review"):
            st.write(cleaned_review)

st.divider()
st.caption("Machine Learning Project | Bernoulli Naïve Bayes")
