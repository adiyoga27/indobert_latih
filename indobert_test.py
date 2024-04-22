import re
import joblib

model_dir = 'assets/models/indobert/'

# Load tokenizer, model, and vectorizer
model = joblib.load(f'{model_dir}sentiment_model.pkl')
vectorizer = joblib.load(f'{model_dir}vectorizer.pkl')


# Preprocessing teks
def clean_text(txt):
    return re.sub(r'[^\w\s]', '', txt)


text = clean_text("Tiktok shop itu bagus banget, aku suka belanja di sana!")

# Tokenize
vectorized = vectorizer.transform([text])

# Predict
sentiment = model.predict(vectorized)

print('Comment:', text)
print('Sentiment:', sentiment[0])
