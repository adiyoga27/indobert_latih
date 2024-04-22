import joblib
import re

from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer

# Load the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

model_dir = 'assets/models/trained/'


# Define a function for text cleaning and lemmatization
def preprocess_text(text):
    # Clean the text
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Lemmatize the words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


# 1. siapkan model dan vectorizer untuk melakukan prediksi

# load model
model = joblib.load(f'{model_dir}sentiment_model.pkl')

# load vectorizer
vectorizer = joblib.load(f'{model_dir}vectorizer.pkl')

# 2. user input
comment = 'aku benci banget sama tiktok shop'

# translate to english
translated = GoogleTranslator(source='id', target='en').translate(preprocess_text(comment))

# 3. vectorize, ubah kalimat menjadi vektor

# vectorize, ubah kalimat menjadi vektor
vectorized = vectorizer.transform([translated])

# predict sentiment
sentiment = model.predict(vectorized)

# 4. print result
print('Comment:', comment)
print('Translated:', translated)
print('\nSentiment:', sentiment[0], vectorized)
