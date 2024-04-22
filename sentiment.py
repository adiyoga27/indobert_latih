import pandas as pd
import numpy as np
import joblib
import nltk
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

export_dir = 'assets/models/'

# download nltk data
if not wordnet.synsets('test'):
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

# load data, pada file kita memiliki field Tanggal & Komentar
data = pd.read_csv('assets/files/data.csv')

# mengambil beberapa data pertama, misalnya 100
# hal ini dilakukan untuk mempercepat proses testing, tapi untuk proses training sebaiknya
# menggunakan sebanyak mungkin data yang ada, hanya saja prosesnya akan lebih lama
data = data.head(2000)

# menyiapkan sastrawi untuk proses stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# menyiapkan wordnet untuk proses lemmatization
lemmatizer = WordNetLemmatizer()

# menyiapkan VADER untuk sentiment analysis
sia = SentimentIntensityAnalyzer()

# menyiapkan stop words
stop_words = set(stopwords.words('english'))


# stemming
def stem_text(text):
    return stemmer.stem(text)


def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


def clean_text(text):
    # Hapus simbol menggunakan ekspresi reguler
    text = re.sub(r'[^\w\s]', '', text)
    return text


def preprocessing(data):
    print('Preprocessing data, please wait...\n')
    with tqdm(total=len(data)) as pbar:
        for i in range(len(data)):
            data.loc[i, 'Komentar'] = clean_text(data.loc[i, 'Komentar'])  # hapus simbol dan karakter aneh

            # CASE FOLDING
            data.loc[i, 'Komentar'] = data.loc[i, 'Komentar'].lower()  # ubah ke huruf kecil
            # data.loc[i, 'Komentar'] = stemmer.stem(data.loc[i, 'Komentar'])  # stemming

            # STEMMING
            data.loc[i, 'Komentar'] = ' '.join([stemmer.stem(word) for word in data.loc[i, 'Komentar'].split()])

            # TOKENIZATION
            # karena library sastrawi kurang akurat dalam proses stemming,
            # kita akan menggunakan lemmatization dari library nltk
            data.loc[i, 'Komentar'] = ' '.join([lemmatizer.lemmatize(word) for word in data.loc[i, 'Komentar'].split()])

            # STOPWORD REMOVAL
            data.loc[i, 'Komentar'] = remove_stopwords(data.loc[i, 'Komentar'])

            pbar.update(1)
    return data


# di sini kita melakukan preprocessing data (cleaning, stemming, lemmatization, etc)
data = preprocessing(data)

# looping untuk menerjemahkan kalimat ke bahasa inggris, karena library textblob hanya support bahasa inggris
# sekaligus kita melakukan sentiment analysis, dan menyimpan hasilnya ke dalam field Sentimen & Label
for index, row in data.iterrows():
    try:
        # terjemahkan kalimat ke bahasa inggris
        translated = GoogleTranslator(source='id', target='en').translate(row['Komentar'])

        # hitung sentiment menggunakan textblob
        # blob = TextBlob(translated)

        # hitung sentiment menggunakan nltk
        vader = sia.polarity_scores(translated)

        # ambil nilai sentiment
        # sentiment = blob.sentiment.polarity
        sentiment = vader['compound']

        # tambahkan sentiment ke dataframe
        data.at[index, 'Sentimen'] = sentiment

        # tambahkan label ke dataframe
        if sentiment > 0:
            data.at[index, 'Label'] = 'Positif'
        elif sentiment < 0:
            data.at[index, 'Label'] = 'Negatif'
        else:
            data.at[index, 'Label'] = 'Netral'

        print('Processing data:', index, 'of', len(data), '(', sentiment, ')', translated)
    except Exception as e:
        data.at[index, 'Sentimen'] = 0
        data.at[index, 'Label'] = 'Netral'

        print('Error processing data at index', index, ':', e)


# FEATURE EXTRACTION -------
# pada tahap ini kita sudah berhasil melakukan sentiment analysis, dan hasilnya disimpan ke dalam field Sentimen & Label
# selanjutnya kita akan melakukan training menggunakan algoritma SVM, langkah ini disebut sebagai "ekstraksi fitur"
# kita perlu mengekstrak fitur dari data teks menggunakan TF-IDF (Term Frequency-Inverse Document Frequency)
# untuk mengubah teks menjadi vektor numerik yang dapat digunakan oleh algoritma SVM

# inisialisasi TF-IDF
vectorizer = TfidfVectorizer()

# ubah teks menjadi vektor
X = vectorizer.fit_transform(data['Komentar'])


# MODEL TRAINING, SVM -------
# pada tahap ini kita akan melakukan training menggunakan algoritma SVM
# SVM digunakan jika kita memiliki data yang tidak terlalu besar, di samping itu SVM tidak membutuhkan
# sumber daya komputasi yang besar dibandingkan ANN, sehingga cocok untuk data yang tidak terlalu besar

# inisialisasi model SVM
svm = SVC()

# latih model SVM
svm.fit(X, data['Label'])

# MODEL EVALUATION -------
# pada tahap ini kita akan melakukan evaluasi model yang sudah kita latih

# prediksi label menggunakan SVM
y_pred_svm = svm.predict(X)

# hitung akurasi
accuracy_svm = np.mean(y_pred_svm == data['Label'])

# tampilkan hasil akurasi
print('Akurasi SVM:', accuracy_svm)
# print(data[['Sentimen', 'Label']])

# SAVE MODEL -------
# setelah model berhasil dilatih, kita perlu menyimpan model ke dalam file
# file yang kita simpan di sini adalah model dan vectorizer yang sudah kita latih
# model digunakan untuk melakukan prediksi atau analisis sentimen, sedangkan vectorizer digunakan untuk
# mengubah teks menjadi vektor numerik yang dapat digunakan oleh algoritma SVM

joblib.dump(svm, export_dir + 'sentiment_model.pkl')
joblib.dump(vectorizer, export_dir + 'vectorizer.pkl')
print('Model saved to ' + export_dir)

# CROSS VALIDATION -------
# kita juga dapat melakukan cross validation untuk mengevaluasi model yang sudah kita latih
# cross validation dilakukan dengan cara membagi data menjadi beberapa bagian, dan melakukan training
# dan testing pada setiap bagian data tersebut

# inisialisasi k-fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# inisialisasi list untuk menyimpan hasil akurasi
accuracies = []

# lakukan cross validation
for train_index, test_index in kfold.split(X, data['Label']):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = data['Label'][train_index], data['Label'][test_index]

    # latih model SVM
    svm.fit(X_train, y_train)

    # prediksi label menggunakan SVM
    y_pred_svm = svm.predict(X_test)

    # hitung akurasi
    accuracy_svm = np.mean(y_pred_svm == y_test)

    # simpan hasil akurasi
    accuracies.append(accuracy_svm)

    print(f'Cross validation accuracy: {kfold.get_n_splits() - len(accuracies) + 1} accuracy: {accuracy_svm:.4f}')

# tampilkan hasil rata-rata akurasi
avg_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy across {len(accuracies)} folds: {avg_accuracy:.4f}")