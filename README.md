<h1 align="center">💼 CareerPath AI — CV ↔️ İş İlanı Eşleşme Uygulaması</h1>

<p align="center">
📊 <b>Veri Bilimi & NLP Projesi</b> • ⚙️ <b>TF-IDF + Cosine Similarity</b> • 🌐 <b>Streamlit Web App</b>
</p>

---

## 🚀 Proje Özeti

**CareerPath AI**, adayların CV’leri ile hedefledikleri iş ilanlarını doğal dil işleme (NLP) teknikleriyle analiz eden bir **Streamlit tabanlı uygulamadır**.  
Uygulama, CV ve iş ilanı metinlerinden becerileri (skills) tespit eder, metin benzerliği hesaplar ve sonunda kullanıcıya:

- 📈 **Eşleşme Skoru (0–100)**  
- 💪 **Güçlü Yönler (Skill Eşleşmeleri)**  
- 🧩 **Geliştirilecek Alanlar (CV’de olmayan ama ilanda istenen beceriler)**  
- ✉️ **Kişiselleştirilmiş Cover Letter (motivasyon mektubu)**  

çıktılarını üretir.

---

## 🎯 Amaç

- CV’nin hedef iş ilanına **ne kadar uygun olduğunu ölçmek**
- Adaylara **somut, beceri bazlı geri bildirim** sunmak
- İşe alım süreçlerinde veri odaklı analizler üretmek
- Klasik NLP tekniklerini gerçek bir iş senaryosunda uygulamak

---

## 🧠 Kullanılan Teknolojiler

| Kategori | Teknoloji / Kütüphane |
|-----------|----------------------|
| Programlama Dili | Python 3.10+ |
| Arayüz | Streamlit |
| NLP | scikit-learn (TF-IDF + Cosine Similarity) |
| Dosya Okuma | PyPDF2, python-docx |
| Veri Ön İşleme | re, string |
| Ortam Yönetimi | venv (virtual environment) |

---

## 🗂️ Proje Klasör Yapısı
```bash
careerpath-ai/
│
├── app/
│   └── main.py          # Ana Streamlit uygulaması
│
├── requirements.txt     # Gerekli kütüphaneler
├── .gitignore           # venv, cache vs. hariç tutma
└── README.md            # Proje dokümantasyonu
```
---

## ⚙️ Kurulum ve Çalıştırma

### 1️⃣ Sanal Ortam Kur
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# veya
source .venv/bin/activate      # macOS / Linux
```
### 2️⃣ Kütüphaneleri Yükle
```bash
pip install -r requirements.txt
```
### 3️⃣ Uygulamayı Başlat
```bash
streamlit run app/main.py
```
Tarayıcıda açılacak: 👉 http://localhost:8501

## 🧩 Uygulama Akışı

| 🔢 Adım | İşlem | Açıklama |
|:-------:|:------|:----------|
| 1️⃣ | **Dosya Yükleme** | Kullanıcı CV ve iş ilanı dosyalarını yükler (PDF, DOCX veya TXT). |
| 2️⃣ | **Metin Ön İşleme** | Metinler temizlenir, normalize edilir, gereksiz karakterler kaldırılır. |
| 3️⃣ | **Skill Çıkarımı** | Lexicon (Python, SQL, Power BI, Tableau vb.) üzerinden beceriler tespit edilir. |
| 4️⃣ | **Benzerlik Hesabı** | TF-IDF + Cosine Similarity + Keyword Overlap ile benzerlik hesaplanır. |
| 5️⃣ | **Sonuç Analizi** | Güçlü yönler ve geliştirilmesi gereken alanlar belirlenir. |
| 6️⃣ | **Cover Letter Üretimi** | Analiz sonuçlarına göre kişiselleştirilmiş motivasyon mektubu oluşturulur. |
| 7️⃣ | **Sonuç Ekranı** | Skor, beceri eşleşmeleri ve mektup kullanıcıya gösterilir. |

---

## 📊 Skor Hesaplama Formülü

Toplam eşleşme skoru üç bileşenden oluşur:  

| 🔹 Bileşen | 🔸 Ağırlık | 🔍 Açıklama |
|:-----------|:----------:|:------------|
| **Skill Coverage** | %50 | İş ilanında geçen becerilerin CV’de bulunma oranı |
| **Text Similarity** | %30 | TF-IDF vektörleri üzerinden hesaplanan metin benzerliği |
| **Keyword Overlap** | %20 | Önemli görev kelimeleri ve anahtar terimlerin örtüşme oranı |

📈 **Formül:**  
\[
\text{Eşleşme Skoru} = 0.50 \times \text{SkillCoverage} + 0.30 \times \text{TextSimilarity} + 0.20 \times \text{KeywordOverlap}
\]

💡 **Örnek:**  
Skill coverage %80, text similarity %70, keyword overlap %50 ise  
→ `0.5*80 + 0.3*70 + 0.2*50 = 71`  
➡️ **Eşleşme Skoru = 71 / 100**

## ✉️ Örnek Çıktı
  - Eşleşme Skoru: 82/100
  - Güçlü Yönler: Python, SQL, Power BI, ETL, Machine Learning
  - Geliştirilecek Alanlar: Git, Docker, Time Series Analysis

Oluşturulan Cover Letter:

I am excited to apply for the Data Scientist position at Habaş…
In recent projects, I built end-to-end data workflows using Python, Power BI, and SQL…
While I’m actively strengthening Git and Docker, I’m a fast learner and collaborative team player.

---

## 🧩 Öne Çıkan Teknik Noktalar
	-	🔤 Metin Temizleme: Boşluk, tire, özel karakter ve “s q l” gibi hatalı ayırımları birleştirir.
	-	🧮 TF-IDF Vectorization: Metin benzerliğini sayısal vektörlerle hesaplar.
	-	🧠 Skill Lexicon: 40’tan fazla teknik beceriyi (Python, SQL, Power BI, Tableau, ETL, Spark, vs.) tanır.
	-	📑 Otomatik Cover Letter: Analiz sonuçlarını doğal bir dille özetler.
	-	🌐 Streamlit Arayüzü: Kullanıcı dostu, etkileşimli bir web arayüzü.

📦 requirements.txt
```bash
streamlit==1.37.0
PyPDF2==3.0.1
python-docx==1.1.2
scikit-learn==1.5.1
python-dotenv==1.0.
```
