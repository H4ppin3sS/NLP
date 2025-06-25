# App.py trên Streamlit
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model
with open("model.pkl","rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

with open("lemmatizer.pkl","rb") as f:
    lemmatizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>','',text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

st.title("🎥 IMDB Sentiment Analysis Demo")

user_review = st.text_area("Enter a review here...")

if st.button("Predict Sentiment"):
    review_clean = clean_text(user_review)
    review_tfidf = vectorizer.transform([review_clean])
    prediction = model.predict(review_tfidf)

    if prediction[0] == 1:
        st.success("✅ Sentiment: Positive 🙂")
    else:
        st.error("❌ Sentiment: Negative 🙁")
