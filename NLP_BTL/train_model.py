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
import matplotlib.pyplot as plt
import seaborn as sns


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


df = pd.read_csv('IMDB-Dataset.csv')
df['clean_review'] = df['review'].apply(clean_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
X = df['clean_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)


X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

print("\n✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=['Negative','Positive']))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'], 
            yticklabels=['Negative','Positive'])
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=df['sentiment'].map({1:'Positive', 0:'Negative'}),
            order=['Positive','Negative'], 
            palette='coolwarm')
plt.title("Distribution of Sentiments in Dataset", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)
plt.show()


feature_names = vectorizer.get_feature_names_out()
word_tfidf_sum = X_train_tfidf.sum(axis=0).A1
words_and_scores = list(zip(feature_names, word_tfidf_sum))
words_and_scores = sorted(words_and_scores, key=lambda x: x[1], reverse=True)

top_words = words_and_scores[:20]
words, scores = zip(*top_words)

plt.figure(figsize=(10,6))
sns.barplot(x=list(scores), y=list(words), color='steelblue')
plt.title("Top 20 Most Important Words in Reviews (Train Set)", fontsize=14)
plt.xlabel("TF-IDF Score Sum", fontsize=12)
plt.ylabel("Word", fontsize=12)
plt.show()


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('lemmatizer.pkl', 'wb') as f:
    pickle.dump(lemmatizer, f)

print("\n✅ Model, Vectorizer, Lemmatizer saved.")
