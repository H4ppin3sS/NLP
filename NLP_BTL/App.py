import pickle
import re
import streamlit as st
from nltk.corpus import stopwords

# Load saved objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('lemmatizer.pkl', 'rb') as f:
    lemmatizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

# Hàm dự đoán
def predict_sentiment(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)
    return "Positive 😀" if prediction[0] == 1 else "Negative 😞"

# ==========================
# Streamlit App
# ==========================
st.title("🎥 Phân tích cảm xúc review phim (IMDB) ")
st.markdown("""
Nhập review bằng tiếng Anh vào ô bên dưới, nhấn nút **Phân tích** 
và xem kết quả dự đoán cảm xúc.
""")

user_input = st.text_area("Nhập review:", "")

if st.button("Phân tích"):
    if user_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập review trước khi phân tích.")
    else:
        result = predict_sentiment(user_input)
        st.markdown(f"### Kết quả: {result}")

