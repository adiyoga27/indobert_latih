import pandas as pd
import numpy as np
import joblib
import nltk
import re
import json
import torch

from tqdm import tqdm

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from transformers import AutoTokenizer, AutoModel

export_dir = 'assets/models/indobert/'

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Allocated to GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")

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
data = data.head(300)

# preprocessing data
# kita membutuhkan tensorflow dan pytorch untuk tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased", model_max_length=256)
factory = StemmerFactory()
stemmer = factory.create_stemmer()
lemmatizer = WordNetLemmatizer()

# menyiapkan stop words
stop_words = set(stopwords.words('english'))

print('Start preprocessing data, please wait...')


def clean_text(text):
    # Hapus simbol menggunakan ekspresi reguler
    text = re.sub(r'[^\w\s]', '', text)
    return text


def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


data['Komentar'] = data['Komentar'].apply(clean_text)

# STEMMING
data['Komentar'] = data['Komentar'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(stemmer.stem(word)) for word in x.split()]))

# REMOVE STOPWORDS
data['Komentar'] = data['Komentar'].apply(remove_stopwords)

# CASE FOLDING
data['Komentar'] = data['Komentar'].str.lower()

json_data = [{"text": comment} for comment in data["Komentar"]]
json_data = [comment["text"] for comment in json_data]

# TOKENIZATION
inputs = tokenizer(json_data, return_tensors='pt', padding=True, truncation=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# analisis sentimen
# tahap ini merupakan tahap dimana kita menganalisis sentimen dari teks yang ada
model = AutoModel.from_pretrained('indolem/indobert-base-uncased')

# sebenarnya kita bisa langsung menulis kode seperti ini, tapi ketika data latihnya besar
# maka proses ini akan memakan waktu yang sangat lama dan tidak ada indikator prosesnya
# outputs = model(input_ids, attention_mask=attention_mask)

# maka dari itu kita akan menggunakan tqdm untuk menampilkan indikator prosesnya
outputs = []
for i in tqdm(range(len(input_ids)), desc="Analyzing sentiment..."):
    ids = input_ids[i].unsqueeze(0)
    mask = attention_mask[i].unsqueeze(0)

    output = model(ids, attention_mask=mask)
    outputs.append(output)

last_hidden_state = torch.cat([output.last_hidden_state for output in outputs], dim=0)
pooler_output = torch.mean(last_hidden_state, dim=1)

# Menggunakan pooler_output untuk prediksi
predictions = np.argmax(pooler_output.detach().numpy(), axis=1)

# pelabelan
# tahap ini merupakan tahap dimana kita memberikan label pada hasil prediksi
data['Sentimen'] = predictions
data['Label'] = data['Sentimen'].apply(lambda x: 'Positif' if x == 1 else 'Negatif' if x == 0 else 'Netral')

# periksa jika semua label tersedia (positif, negatif, netral)
labels = data['Label'].unique()

# memastikan semua label (class) tersedia, sebab ketika data latih yang kita gunakan sedikit (untuk testing)
# kemungkinan ada class yang tidak muncul, dan itu dapat menyebabkan error
unlabels = ['Positif', 'Negatif', 'Netral']
for unlabel in unlabels:
    if unlabel not in labels:
        new_row = pd.DataFrame({'Label': [unlabel], 'Sentimen': 0, 'Komentar': 'dummy'})
        data = pd.concat([data, new_row], ignore_index=True)

# untuk memeriksa jika data yang kita tambahkan sudah benar (uncomment jika tidak dibutuhkan)
# data.to_json('assets/outputs/indobert-output-data-sample.json', orient='records', indent=4)

# fitur ekstraksi
# ini merupakan proses mengubah data teks menjadi representasi numerik yang dapat dimengerti oleh model machine learning
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Komentar'])

# training model
# tahap ini merupakan tahap dimana kita melatih model machine learning kita
X_train, X_test, y_train, y_test = train_test_split(X, data['Label'], test_size=0.2)
svm = SVC()
svm.fit(X_train, y_train)

# evaluasi model
# tahap ini sangat penting untuk mengetahui seberapa baik model yang kita buat
y_pred_svm = svm.predict(X_test)
accuracy_svm = np.mean(y_pred_svm == y_test)
print('SVM accuracy:', accuracy_svm)

# save model
# tahap ini merupakan tahap dimana kita menyimpan model yang sudah kita buat
joblib.dump(svm, f'{export_dir}sentiment_model.pkl')
joblib.dump(vectorizer, f'{export_dir}vectorizer.pkl')

# cross validation
# tahap ini merupakan tahap dimana kita melakukan cross validation untuk mengevaluasi model kita
skf = StratifiedKFold(n_splits=5)
accuracies = []

for train_index, test_index in skf.split(X, data['Label']):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = data['Label'][train_index], data['Label'][test_index]

    svm = SVC()
    try:
        svm.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during cross-validation: {e}")
        continue

    y_pred_svm = svm.predict(X_test)
    accuracy = np.mean(y_pred_svm == y_test)
    accuracies.append(accuracy)

    print(f'Cross validation accuracy: {skf.get_n_splits() - len(accuracies) + 1} accuracy: {accuracy:.4f}')

# tampilkan hasil rata-rata akurasi
avg_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy across {len(accuracies)} folds: {avg_accuracy:.4f}")
