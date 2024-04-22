import re
import nltk
import torch

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from transformers import BertTokenizer, AutoModel

print('Start preprocessing data, please wait...')


def clean_text(text):
    # Hapus simbol menggunakan ekspresi reguler
    text = re.sub(r'[^\w\s]', '', text)
    return text


# Definisikan fungsi untuk stemming dan lemmatization
def stem_and_lemmatize(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(stemmer.stem(word)) for word in text.split()])


# download nltk data
if not wordnet.synsets('test'):
    nltk.download('wordnet')
    nltk.download('vader_lexicon')

# Definisikan tokenizer IndoBERT
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Definisikan model IndoBERT
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")


# Definisikan model klasifikasi sentimen (contoh)
class SentimentClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 3)

    def forward(self, x):
        logits = self.linear(x)
        return logits


def analyst(sentence):
    # Preprocessing teks
    sentence = clean_text(sentence)
    sentence = stem_and_lemmatize(sentence)

    # Tokenisasi teks
    inputs = torch.LongTensor(tokenizer.encode(sentence)).view(1, -1)

    # Dapatkan output model IndoBERT
    outputs = model(inputs)

    # Dapatkan representasi kalimat
    sentence_embedding = outputs.last_hidden_state[:, 0, :]

    # Klasifikasikan sentimen
    sentiment_model = SentimentClassifier()
    sentiment_logits = sentiment_model(sentence_embedding)
    sentiment_probs = torch.nn.functional.softmax(sentiment_logits, dim=1)

    # Interpretasi hasil
    predicted_sentiment = torch.argmax(sentiment_probs, dim=1)
    sentiment_label = 'Netral'

    if predicted_sentiment == 0:
        sentiment_label = 'Positif'
    elif predicted_sentiment == 2:
        sentiment_label = 'Negatif'

    return {
        'sentiment': sentiment_label,
        'probabilities': sentiment_probs.tolist()
    }


comment = 'tiktok shop sangat baik pelayanannya di Indonesia Aku sangat suka, Aku harap mereka tetap di Indonesia!'

print(f'Kalimat: {comment}')
print(f'Sentimen: {analyst(comment)["sentiment"]}')
print(f'Probabilitas: {analyst(comment)["probabilities"]}')
