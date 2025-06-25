import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)           # Xóa thẻ HTML
    text = re.sub(r'[^a-zA-Z]', ' ', text)      # Xóa ký tự đặc biệt
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


st.title("🎥 IMDB Sentiment Analysis Demo (Logistic Regression)")


if st.button("Train Model"):
    with st.spinner("Đang huấn luyện mô hình..."):
        df = pd.read_csv('IMDB-Dataset.csv')
        df['clean_review'] = df['review'].apply(clean_text)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        X = df['clean_review']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)

        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train_tfidf, y_train)

        # Save model, vectorizer, lemmatizer
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        with open('lemmatizer.pkl', 'wb') as f:
            pickle.dump(lemmatizer, f)

        # Evaluate
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"✅ **Accuracy**: {accuracy:.4f}")

        st.text("\nClassification Report:\n" + str(classification_report(y_test, y_pred, target_names=['Negative','Positive'])))

        st.write("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        st.success("✅ Model trained and saved!")


model = None
vectorizer = None

if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl") and os.path.exists("lemmatizer.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("lemmatizer.pkl", "rb") as f:
        lemmatizer = pickle.load(f)


st.subheader("🔍 Test a Review")

user_review = st.text_area("Enter a review here...")

if st.button("Predict Sentiment"):
    if model and vectorizer:
        review_clean = clean_text(user_review)
        review_tfidf = vectorizer.transform([review_clean])
        prediction = model.predict(review_tfidf)

        if prediction[0] == 1:
            st.success("✅ Sentiment: Positive 🙂")
        else:
            st.error("❌ Sentiment: Negative 🙁")
    else:
        st.warning("⚠️ Model or Vectorizer not found. Please train or provide the model files first.")
