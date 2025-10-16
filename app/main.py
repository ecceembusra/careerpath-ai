"""
CareerPath AI — CV ↔️ İş İlanı Eşleşme (Streamlit)
---------------------------------------------------
Bu uygulama, kullanıcının CV'si ile seçilen iş ilanı arasındaki uyumu analiz eder:
1) Metin ön-işleme ve normalizasyon
2) Skill çıkarımı (lexicon + alias normalizasyonu)
3) Üç bileşenli skor:
   - %50 Skill Coverage   : JD'de istenen skill'lerin CV'de bulunma oranı
   - %30 Text Similarity  : TF-IDF + Cosine benzerliği
   - %20 Keyword Overlap  : Görev/anahtar sözcük ortaklığı
4) Çıktılar: Eşleşme skoru, güçlü/geliştirilecek alanlar, uzun cover letter
"""

# ============== 1) KÜTÜPHANELER ==============
import io, re, textwrap, string
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============== 2) STREAMLIT SAYFA AYARI ==============
st.set_page_config(page_title="CareerPath AI", page_icon="💼", layout="wide")
st.title("CareerPath AI — CV ↔️ İş İlanı Eşleşme")


# ============== 3) SÖZLÜKLER (SKILL & KEYWORD) ==============
# - SKILL_LEXICON: Uygulamanın doğrudan metinde aradığı teknik beceriler
# - JD_KEYWORDS  : Görev/anahtar sözcükler (pipeline, dashboard vb.) — skoru dengeler
# - ALIASES      : Yazım varyasyonlarını tekilleştirme (ms sql -> sql, a/b testing -> ab testing gibi)

SKILL_LEXICON = [
    # Core / DB
    "python","pandas","numpy","sql","postgresql","mysql","sql server","t-sql","ms sql","bigquery",
    # BI & Viz
    "power bi","tableau","looker","google data studio","data visualization","excel","gsheet",
    # ML
    "scikit-learn","machine learning","deep learning","nlp","tensorflow","pytorch",
    "xgboost","random forest","time series","ab testing","a/b testing",
    # Pipelines & DWH
    "etl","ssis","ssas","ssrs","airflow","data modeling","data warehouse",
    # Big Data & Cloud & Tooling
    "spark","hadoop","aws","azure","google cloud","git","github","docker","marketing analytics",
]

JD_KEYWORDS = [
    "dashboard","reporting","pipeline","modeling","deployment","ab testing",
    "segmentation","forecast","recommendation","churn","feature engineering",
    "kpi","optimization","automation","stakeholder","storytelling","experiment",
    "etl","sql","python","power bi","looker","tableau","spark","hadoop","bigquery",
    "airflow","docker","git","time series","marketing analytics"
]

ALIASES = {
    "ms sql":"sql", "mssql":"sql", "sqlserver":"sql", "t sql":"t-sql",
    "google bigquery":"bigquery", "gcp":"google cloud",
    "a b testing":"ab testing", "a/b testing":"ab testing",
}


# ============== 4) YARDIMCI FONKSİYONLAR ==============

# ---- 4.1 Dosya okuyucular ----
def read_pdf(file_bytes: bytes) -> str:
    """PDF dosyasından sayfa sayfa metin çıkarır."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def read_docx(file_bytes: bytes) -> str:
    """DOCX içindeki paragrafları birleştirir."""
    f = io.BytesIO(file_bytes)
    doc = Document(f)
    return "\n".join([p.text for p in doc.paragraphs])

def read_any(file) -> str:
    """Yüklenen dosyanın uzantısına göre uygun okuyucuyu çağırır."""
    name = (file.name or "").lower()
    content = file.read()
    if name.endswith(".pdf"):
        return read_pdf(content)
    if name.endswith(".docx"):
        return read_docx(content)
    # TXT ve diğer durumlar
    return content.decode("utf-8", errors="ignore")


# ---- 4.2 Metin normalizasyonu ----
def _normalize_simple(t: str) -> str:
    """
    Temel normalizasyon:
    - Küçük harfe çevir
    - 's q l' gibi ayrık yazımları birleştir
    - 't-sql' varyasyonlarını tekilleştir
    - tire ve noktalama kaldır, fazla boşlukları sadeleştir
    """
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r"s\W*q\W*l", "sql", t)       # s q l -> sql
    t = re.sub(r"t\W*-\W*sql", "t-sql", t)   # t - sql -> t-sql
    t = t.replace("-", " ")
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_text(t: str) -> str:
    """Kullanım kolaylığı için tek satırlık sarmalayıcı."""
    return _normalize_simple(t)


# ---- 4.3 Skill çıkarımı ----
def extract_skills(text: str) -> list[str]:
    """
    CV/İlan metninden sözlük bazlı skill çıkarımı.
    1) Metni normalize et
    2) Alias/varyasyonları kanonik isimlere çevir
    3) SKILL_LEXICON'daki hedef terimleri ara
    """
    txt = " " + _normalize_simple(text) + " "

    # Metin içinde alias normalizasyonu yap (örn: a/b testing -> ab testing)
    for k, v in ALIASES.items():
        txt = txt.replace(" " + _normalize_simple(k) + " ", " " + _normalize_simple(v) + " ")

    # Sözlük tabanlı tarama
    hits = set()
    for s in SKILL_LEXICON:
        s_norm = " " + _normalize_simple(s) + " "
        if s_norm in txt:
            hits.add(ALIASES.get(_normalize_simple(s), _normalize_simple(s)))  # kanonik isim
    return sorted(hits)


# ---- 4.4 Benzerlik ve skor fonksiyonları ----
def tfidf_cosine(a: str, b: str) -> float:
    """TF-IDF + Cosine ile 0–1 arası metin benzerliği skorunu döndürür."""
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    X = vect.fit_transform([a or "", b or ""])
    return float(cosine_similarity(X[0], X[1])[0,0])

def keyword_overlap_score(cv_text: str, jd_text: str) -> float:
    """
    JD_KEYWORDS listesindeki anahtarların hem CV hem JD'de geçme oranı.
    0–1 arası skor döner (normalize).
    """
    cv_n = " " + _normalize_simple(cv_text) + " "
    jd_n = " " + _normalize_simple(jd_text) + " "
    hits, used = 0, 0
    for kw in JD_KEYWORDS:
        used += 1
        k = " " + _normalize_simple(kw) + " "
        if (k in cv_n) and (k in jd_n):
            hits += 1
    return 0.0 if used == 0 else hits / used

def compute_match_score(cv_text: str, jd_text: str, cv_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Nihai skor (0–100) = 0.50*skill_coverage + 0.30*text_sim + 0.20*keyword_overlap
    Ayrıntılı kırılımı da döndürür.
    """
    # 1) Skill coverage: JD'de istenen skill'lerin CV'de bulunma oranı
    cov = (len(set(cv_skills) & set(jd_skills)) / max(1, len(jd_skills)))
    # 2) Metin benzerliği: TF-IDF cosine
    sim = tfidf_cosine(cv_text, jd_text)
    # 3) Anahtar kelime örtüşmesi
    kw  = keyword_overlap_score(cv_text, jd_text)

    score = (0.50 * cov + 0.30 * sim + 0.20 * kw) * 100
    score = max(0, min(100, round(score)))  # 0–100 aralığına sıkıştır

    return {
        "score": score,
        "breakdown": {
            "skill_coverage": round(cov * 100),
            "text_similarity": round(sim * 100),
            "keyword_overlap": round(kw  * 100),
        },
    }


# ---- 4.5 Cover letter üretimi ----
def make_cover_letter(cv_skills: list[str], gaps: list[str], role: str, company: str,
                      tone: str = "professional", words: int = 220) -> str:
    """
    CV-JD analizine göre uzun ve motive edici bir cover letter oluşturur.
    'tone' ve 'words' parametreleri ile stil/uzunluk ayarı yapılabilir.
    """
    strengths = ", ".join((cv_skills or [])[:10]) if cv_skills else "relevant skills"
    soften    = ", ".join((gaps or [])[:6]) if gaps else "no major gaps"

    opening = {
        "professional": f"I am excited to apply for the {role} position at {company}. I combine hands-on analytics with clear communication and an ownership mindset.",
        "friendly":     f"I'm thrilled to apply for the {role} role at {company}! I love turning messy data into useful, human-readable insights."
    }[tone]

    body1 = (
        f"In recent projects, I built end-to-end data workflows using {strengths}. "
        f"I collaborated with cross-functional teams to define KPIs, designed reliable data models, and automated reporting. "
        f"I value clean data practices, reproducible code, and measurable impact."
    )

    body2 = "I learn fast and enjoy tackling ambiguous problems. "
    if gaps and gaps != ["no major gaps"]:
        body2 += f"I'm actively strengthening {soften}, and I approach gaps with a clear learning plan and rapid prototyping."

    closing = (
        f"Thank you for your time and consideration. "
        f"I would welcome the chance to discuss how I can contribute to {company}'s roadmap."
    )

    # Basit kelime sayısı kontrolü (cut-off)
    txt = " ".join([opening, body1, body2, closing]).strip()
    if words and words > 0:
        target = int(words)
        tokens = txt.split()
        if len(tokens) > target:
            txt = " ".join(tokens[:target]) + "..."
    return txt


# ============== 5) ARAYÜZ (UI) ==============

# ---- 5.1 Dosya yükleme alanları ----
col1, col2 = st.columns(2)
with col1:
    cv_file = st.file_uploader("CV yükle (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="cv")
with col2:
    jd_file = st.file_uploader("İş ilanı yükle (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="jd")

# ---- 5.2 Kullanıcı ayarları ----
role    = st.text_input("Pozisyon başlığı", value="Data Scientist")
company = st.text_input("Şirket adı", value="Your Company")
tone    = st.selectbox("Cover letter tonu", ["professional","friendly"], index=0)
words   = st.slider("Cover letter uzunluğu (kelime)", 120, 300, 220, 10)

# ---- 5.3 Hesapla butonu ----
if st.button("Eşleşmeyi Hesapla", type="primary"):

    # 1) Dosya kontrolü
    if not (cv_file and jd_file):
        st.warning("Lütfen hem CV hem de iş ilanı dosyasını yükleyin.")
        st.stop()

    # 2) Metinleri oku + normalize et
    cv_text = clean_text(read_any(cv_file))
    jd_text = clean_text(read_any(jd_file))

    # 3) Skill çıkarımı
    cv_skills = extract_skills(cv_text)
    jd_skills = extract_skills(jd_text)

    # 4) Skor hesapla + güçlü / geliştirilmesi gereken alanlar
    result     = compute_match_score(cv_text, jd_text, cv_skills, jd_skills)
    strengths  = sorted(set(cv_skills) & set(jd_skills))
    gaps       = sorted(set(jd_skills) - set(cv_skills))

    # 5) Sonuçlar
    st.metric("Eşleşme Skoru", f"{result['score']}/100")
    st.caption(
        f"🧮 Kırılım → Skills: {result['breakdown']['skill_coverage']} | "
        f"TF-IDF: {result['breakdown']['text_similarity']} | "
        f"Keywords: {result['breakdown']['keyword_overlap']}"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Güçlü Yönler (Skill Eşleşmeleri)**")
        st.write(strengths or "—")
    with c2:
        st.markdown("**Geliştirilecek Alanlar (JD’de var, CV’de yok)**")
        st.write(gaps or "—")

    # (İsteğe bağlı debug çıktısı)
    # st.caption(f"cv_skills: {cv_skills}")
    # st.caption(f"jd_skills: {jd_skills}")

    # 6) Cover Letter
    st.subheader("✉️ Cover Letter Taslağı")
    letter = make_cover_letter(strengths or cv_skills, gaps, role or "the role", company or "your company",
                               tone=tone, words=words)
    st.write(letter)
