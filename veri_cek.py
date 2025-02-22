import requests
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# VADER lexicon'u indir
nltk.download('vader_lexicon')

# Sentiment analizcisini başlat
sia = SentimentIntensityAnalyzer()

# Stack Overflow API URL'si
url = "https://api.stackexchange.com/2.3/questions"

# API'ye gönderilecek parametreler
params = {
    'order': 'desc',  # Soruları yeniye göre sıralar
    'sort': 'activity',  # En aktif sorular
    'tagged': 'python',  # Python etiketiyle ilgili soruları alır
    'site': 'stackoverflow',  # Stack Overflow sitesinden veri çeker
    'pagesize': 100  # API’nin maksimum çekebileceği veri sayısı
}

# 1500+ veri çekmek için sayfalama ekleyelim
questions = []
page = 1

while len(questions) < 1500:
    params['page'] = page  # Her sayfa için güncelle
    response = requests.get(url, params=params)

    # Hata kontrolü
    if response.status_code != 200:
        print(f"API isteği başarısız! Hata kodu: {response.status_code}")
        break

    data = response.json()

    # API verisi boş mu kontrol et
    if 'items' not in data or not data['items']:
        print("Veri bitti veya çekilemedi!")
        break

    # Veriyi listeye ekle
    questions.extend(data['items'])
    page += 1  # Bir sonraki sayfaya geç

    # API limitlerine takılmamak için
    if page > 15:  # 100 x 15 = 1500 veri olacak
        break

# 1500 veriden fazla çekildiyse sınırla
questions = questions[:1500]

# Veriyi DataFrame'e çevir
df = pd.DataFrame(questions)

print(f"Toplam {len(df)} veri çekildi.")

# Sentiment analizi ekleyerek yeni bir sütun oluştur
df['sentiment'] = df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# İlk 10 soruyu yazdır (kontrol amaçlı)
for index, row in df.head(10).iterrows():
    print(f"Title: {row['title']}")
    print(f"Link: {row['link']}")
    print(f"Score: {row['score']}")
    print(f"Sentiment Score: {row['sentiment']}")
    print("-" * 40)

# Sentiment skorlarına göre çubuk grafik çiz
plt.figure(figsize=(12, 6))
plt.bar(df['title'][:20], df['sentiment'][:20], color=['green' if s > 0 else 'red' for s in df['sentiment'][:20]])
plt.xticks(rotation=45, ha='right')
plt.xlabel('Başlıklar')
plt.ylabel('Sentiment Skoru')
plt.title('Stack Overflow Soruları Sentiment Skorları (İlk 20)')
plt.tight_layout()

# Grafiği göster
plt.show()
