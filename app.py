import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import StringIO

# =====================================================================
# Konfigurasi Halaman dan Header
# =====================================================================
st.set_page_config(page_title="Klasifikasi Berita", layout="wide")
st.markdown("<h3 style='text-align: left; color: gray;'>Kelompok 3: Klasifikasi Berita</h3>", unsafe_allow_html=True)

# =====================================================================
# Load Model dan Vectorizer
# =====================================================================
try:
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("File model atau vectorizer tidak ditemukan. Pastikan file .pkl ada di direktori yang sama.")
    st.stop()


# =====================================================================
# Sidebar
# =====================================================================
menu = st.sidebar.radio("📚 Menu", ["Klasifikasi", "Tentang"])
lang = st.sidebar.selectbox("🌐 Pilih Bahasa | Choose Language", ["Indonesia", "English"])


# === Tentang Aplikasi ===
if menu == "Tentang":
    if lang == "Indonesia":
        st.title("ℹ️ Tentang Aplikasi")
        st.write("""
Kelompok 3 dalam proyek ini membangun sistem klasifikasi berita menggunakan tiga modul utama: Regresi Logistik, Random Forest, dan Naive Bayes. Tujuan utama aplikasi ini adalah membantu pengguna mengidentifikasi apakah suatu berita termasuk kategori **REAL** atau **FAKE** dengan menggunakan teknik pembelajaran mesin (machine learning).
        """)

        st.markdown("---")
        st.subheader("🧠 Modul yang Digunakan")

        st.subheader("1. Regresi Logistik")
        st.write("""
Regresi Logistik adalah algoritma klasifikasi yang populer digunakan untuk masalah biner. Model ini mencoba memetakan input ke dalam probabilitas dua kelas berbeda, menggunakan fungsi logistik. Meskipun sederhana, regresi logistik sering memberikan hasil yang kompetitif dalam teks klasifikasi.

**Classification Report - Logistic Regression:**

```
               precision    recall  f1-score   support

           0       0.50      0.48      0.49       157
           1       0.45      0.47      0.46       143

    accuracy                           0.48       300
   macro avg       0.48      0.48      0.48       300
weighted avg       0.48      0.48      0.48       300
```

**Accuracy Score - Logistic Regression: 0.4767**
        """)

        st.subheader("2. Random Forest")
        st.write("""
Random Forest adalah metode ensemble yang menggunakan banyak pohon keputusan untuk meningkatkan akurasi prediksi dan mengurangi risiko overfitting. Model ini sangat cocok untuk menangani data teks yang bervariasi.

**Classification Report - Random Forest:**

```
              precision    recall  f1-score   support

           0       0.55      0.63      0.59       157
           1       0.52      0.43      0.47       143

    accuracy                           0.54       300
   macro avg       0.53      0.53      0.53       300
weighted avg       0.53      0.54      0.53       300
```

**Confusion Matrix:**
```
[[99 58]
 [81 62]]
```

**Accuracy Score - Random Forest: 0.5367**
        """)

        st.subheader("3. Naive Bayes")
        st.write("""
Naive Bayes adalah algoritma probabilistik berdasarkan Teorema Bayes dengan asumsi independensi antar fitur. Dalam klasifikasi teks, meskipun sederhana dan asumsi independensinya jarang terpenuhi, algoritma ini dikenal cepat dan efisien.

**Classification Report - Naive Bayes:**

```
              precision    recall  f1-score   support

           0       0.51      0.69      0.59       157
           1       0.44      0.27      0.34       143

    accuracy                           0.49       300
   macro avg       0.48      0.48      0.46       300
weighted avg       0.48      0.49      0.47       300
```

**Accuracy Score - Naive Bayes: 0.49**
        """)

        st.markdown("---")
        st.subheader("🧾 Kesimpulan")
        st.write("""
Berdasarkan hasil evaluasi yang diperoleh dari ketiga model, dapat disimpulkan bahwa **Random Forest** memiliki performa paling baik dengan akurasi tertinggi sebesar **53.67%**, diikuti oleh **Naive Bayes** dengan **49%**, dan **Logistic Regression** dengan **47.67%**. 

Walaupun semua model menunjukkan hasil akurasi di bawah 60%, ini bisa menjadi indikasi bahwa data yang digunakan memiliki tantangan tersendiri, seperti distribusi yang tidak seimbang atau kurangnya fitur penting yang membedakan antara berita asli dan palsu. Untuk peningkatan ke depan, perlu dipertimbangkan penggunaan dataset yang lebih besar, fitur engineering yang lebih mendalam. Eksperimen ini tetap memberikan wawasan berharga terhadap bagaimana berbagai model machine learning bekerja dalam konteks klasifikasi berita.
        """)

    else: # English version
        st.title("ℹ️ About the App")
        st.write("""
In this project, Group 3 has developed a news classification system using three main modules: Logistic Regression, Random Forest, and Naive Bayes. The primary goal of this application is to assist users in identifying whether a news article is **REAL** or **FAKE** using machine learning techniques.
        """)

        st.markdown("---")
        st.subheader("🧠 Modules Used")

        st.subheader("1. Logistic Regression")
        st.write("""
Logistic Regression is a popular classification algorithm used for binary problems. This model attempts to map inputs into the probability of two distinct classes using a logistic function. Despite its simplicity, logistic regression often yields competitive results in text classification.

**Classification Report - Logistic Regression:**

```
               precision    recall  f1-score   support

           0       0.50      0.48      0.49       157
           1       0.45      0.47      0.46       143

    accuracy                           0.48       300
   macro avg       0.48      0.48      0.48       300
weighted avg       0.48      0.48      0.48       300
```

**Accuracy Score - Logistic Regression: 0.4767**
        """)

        st.subheader("2. Random Forest")
        st.write("""
Random Forest is an ensemble method that utilizes multiple decision trees to enhance prediction accuracy and mitigate the risk of overfitting. It is particularly well-suited for handling diverse text data.

**Classification Report - Random Forest:**

```
              precision    recall  f1-score   support

           0       0.55      0.63      0.59       157
           1       0.52      0.43      0.47       143

    accuracy                           0.54       300
   macro avg       0.53      0.53      0.53       300
weighted avg       0.53      0.54      0.53       300
```

**Confusion Matrix:**
```
[[99 58]
 [81 62]]
```

**Accuracy Score - Random Forest: 0.5367**
        """)

        st.subheader("3. Naive Bayes")
        st.write("""
Naive Bayes is a probabilistic algorithm based on Bayes' Theorem, with the assumption of independence between features. In text classification, despite its simplicity and the fact that its independence assumption is rarely fully met, the algorithm is known for its speed and efficiency.

**Classification Report - Naive Bayes:**

```
              precision    recall  f1-score   support

           0       0.51      0.69      0.59       157
           1       0.44      0.27      0.34       143

    accuracy                           0.49       300
   macro avg       0.48      0.48      0.46       300
weighted avg       0.48      0.49      0.47       300
```

**Accuracy Score - Naive Bayes: 0.49**
        """)

        st.markdown("---")
        st.subheader("🧾 Conclusion")
        st.write("""
Based on the evaluation results from the three models, it can be concluded that **Random Forest** exhibits the best performance with the highest accuracy of **53.67%**, followed by **Naive Bayes** at **49%**, and **Logistic Regression** at **47.67%**.

Although all models show accuracy results below 60%, this may indicate that the dataset presents unique challenges, such as an imbalanced distribution or a lack of critical features distinguishing between real and fake news. For future enhancements, considering a larger dataset and more in-depth feature engineering would be beneficial. This experiment nonetheless provides valuable insights into how various machine learning models perform in the context of news classification.
        """)

# =====================================================================
# Halaman "Klasifikasi" (Teks Tunggal & Unggah File)
# =====================================================================
elif menu == "Klasifikasi":
    # --- Bagian Klasifikasi Teks Tunggal ---
    if lang == "Indonesia":
        st.title("📰 Klasifikasi Berita: Real atau Fake?")
        st.write("Masukkan judul dan isi berita untuk memprediksi apakah berita tersebut benar atau palsu.")
        btn_example = "Gunakan Contoh Berita"
        label_title = "Judul Berita"
        label_text = "Isi Berita"
        btn_predict = "Prediksi Teks"
        warning_input = "Silakan masukkan judul atau isi berita."
        label_real = "✅ Berita ini terdeteksi sebagai: REAL"
        label_fake = "❌ Berita ini terdeteksi sebagai: FAKE"
    else: # English
        st.title("📰 News Classification: Real or Fake?")
        st.write("Enter the news title and content to predict whether it's real or fake.")
        btn_example = "Use Example News"
        label_title = "News Title"
        label_text = "News Content"
        btn_predict = "Predict Text"
        warning_input = "Please enter a title or content first."
        label_real = "✅ This news is detected as: REAL"
        label_fake = "❌ This news is detected as: FAKE"

    if st.button(btn_example):
        st.session_state.title = "Government announces new subsidies for SMEs"
        st.session_state.text = (
            "The government today announced a new subsidy program to support micro, small, and medium enterprises..."
        )
        st.rerun()

    title = st.text_input(label_title, value=st.session_state.get("title", ""))
    text = st.text_area(label_text, value=st.session_state.get("text", ""))

    if st.button(btn_predict):
        if not title.strip() and not text.strip():
            st.warning(warning_input)
        else:
            input_text = title + " " + text
            vectorized_input = vectorizer.transform([input_text])
            prediction = model.predict(vectorized_input)[0]
            
            if prediction == 1:
                st.success(label_real)
            else:
                st.error(label_fake)
    
    st.markdown("---")

    # --- Bagian Unggah & Klasifikasi File ---
    if lang == "Indonesia":
        st.header("📂 Atau Unggah File CSV untuk Klasifikasi Massal")
        st.info("Pastikan file Anda memiliki kolom 'title' dan 'text' untuk hasil terbaik.")
    else: # English
        st.header("📂 Or Upload a CSV File for Bulk Classification")
        st.info("Ensure your file has 'title' and 'text' columns for the best results.")

    uploaded_file = st.file_uploader("Pilih file CSV", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            if 'title' in df_uploaded.columns and 'text' in df_uploaded.columns:
                if not pd.api.types.is_string_dtype(df_uploaded['title']) or not pd.api.types.is_string_dtype(df_uploaded['text']):
                    if lang == "Indonesia":
                        st.error("❌ **Peringatan:** Kolom 'title' dan 'text' harus berisi data teks (kata/kalimat), bukan angka.")
                    else:
                        st.error("❌ **Warning:** The 'title' and 'text' columns must contain text data (words/sentences), not numbers.")
                else:
                    st.success("✅ File berhasil diunggah. Klik tombol di bawah untuk memulai klasifikasi.")
                    
                    if st.button("Mulai Klasifikasi File" if lang == "Indonesia" else "Start File Classification", type="primary"):
                        with st.spinner('Memproses file...'):
                            df_uploaded['full_text'] = df_uploaded['title'].astype(str) + " " + df_uploaded['text'].astype(str)
                            X_uploaded = vectorizer.transform(df_uploaded['full_text'])
                            predictions = model.predict(X_uploaded)
                            df_uploaded['prediksi'] = ['REAL' if p == 1 else 'FAKE' for p in predictions]
                            
                            st.write("**Hasil Klasifikasi:**")
                            st.dataframe(df_uploaded)
                            
                            csv = df_uploaded.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Unduh Hasil Klasifikasi" if lang == "Indonesia" else "Download Classification Results",
                                data=csv,
                                file_name='hasil_klasifikasi.csv',
                                mime='text/csv',
                            )
            else:
                if lang == "Indonesia":
                    st.error("❌ **Peringatan:** File CSV Anda harus memiliki kolom bernama 'title' dan 'text'.")
                else:
                    st.error("❌ **Warning:** Your CSV file must contain columns named 'title' and 'text'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

# === Creator Display (Footer) ===
creator_info_html = """
<div style="position: fixed; bottom: 20px; right: 20px; background-color: #f0f2f6; padding: 10px 15px; border-radius: 5px; box-shadow: 0px 0px 10px #888888; z-index: 1000; text-align: left;">
    <h5 style='color: #333; margin-bottom: 5px;'>👨‍💻 Kelompok 3</h5>
    <p style='color: #333; margin: 0;'>Tegar Maulana Putra</p>
    <p style='color: #333; margin: 0;'>Kayla Puspita Khairiyah</p>
</div>
"""
st.markdown(creator_info_html, unsafe_allow_html=True)
