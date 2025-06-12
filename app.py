import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree
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


# =====================================================================
# Halaman "Tentang Aplikasi"
# =====================================================================
if menu == "Tentang":
    if lang == "Indonesia":
        st.title("ℹ️ Tentang Aplikasi")
        st.write("Penjelasan mengenai proyek, modul yang digunakan, dan hasil evaluasi model ditempatkan di sini.")
    else: # English version
        st.title("ℹ️ About the App")
        st.write("An explanation of the project, the modules used, and the model evaluation results are placed here.")

# =====================================================================
# Halaman "Klasifikasi" (Gabungan Teks Tunggal & Unggah File)
# =====================================================================
elif menu == "Klasifikasi":
    # --- Bagian Klasifikasi Teks Tunggal ---
    if lang == "Indonesia":
        st.title("📰 Klasifikasi Berita: Real atau Fake?")
        st.write("Masukkan judul dan isi berita di bawah ini untuk memprediksi apakah berita tersebut benar atau palsu.")
        btn_predict = "Prediksi Teks"
    else: # English
        st.title("📰 News Classification: Real or Fake?")
        st.write("Enter the news title and content below to predict whether it is real or fake.")
        btn_predict = "Predict Text"

    title = st.text_input("Judul Berita" if lang == "Indonesia" else "News Title")
    text = st.text_area("Isi Berita" if lang == "Indonesia" else "News Content", height=200)

    if st.button(btn_predict):
        if not title.strip() and not text.strip():
            st.warning("Silakan masukkan judul atau isi berita." if lang == "Indonesia" else "Please enter a title or content.")
        else:
            input_text = title + " " + text
            vectorized_input = vectorizer.transform([input_text])
            prediction = model.predict(vectorized_input)[0]
            
            if prediction == 1:
                st.success("✅ Berita ini terdeteksi sebagai: REAL" if lang == "Indonesia" else "✅ This news is detected as: REAL")
            else:
                st.error("❌ Berita ini terdeteksi sebagai: FAKE" if lang == "Indonesia" else "❌ This news is detected as: FAKE")
    
    st.markdown("---")

    # --- Bagian Unggah & Klasifikasi File Eksternal ---
    if lang == "Indonesia":
        st.header("📂 Atau Unggah File Berita Eksternal (.csv)")
        st.info("Fitur ini memungkinkan Anda untuk mengklasifikasikan banyak berita sekaligus dari file CSV Anda. Pastikan file memiliki kolom 'title' dan 'text'.")
    else: # English
        st.header("📂 Or Upload an External News File (.csv)")
        st.info("This feature allows you to classify multiple news articles at once from your CSV file. Ensure the file has 'title' and 'text' columns.")

    uploaded_file = st.file_uploader("Pilih file CSV Anda", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            if 'title' in df_uploaded.columns and 'text' in df_uploaded.columns:
                st.success("✅ File berhasil diunggah. Klik tombol di bawah untuk memulai analisis.")
                
                if st.button("Mulai Klasifikasi File" if lang == "Indonesia" else "Start File Classification", type="primary"):
                    with st.spinner('Menganalisis file Anda...' if lang == "Indonesia" else 'Analyzing your file...'):
                        
                        df_uploaded['full_text'] = df_uploaded['title'].astype(str).fillna('') + " " + df_uploaded['text'].astype(str).fillna('')
                        X_uploaded = vectorizer.transform(df_uploaded['full_text'])
                        predictions = model.predict(X_uploaded)
                        df_uploaded['prediksi'] = ['REAL' if p == 1 else 'FAKE' for p in predictions]
                        
                        st.markdown("---")
                        st.subheader("Hasil Klasifikasi File Anda")
                        st.dataframe(df_uploaded)

                        # Visualisasi Hasil
                        st.subheader("📊 Ringkasan Visual")
                        label_counts = df_uploaded['prediksi'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Jumlah Prediksi**")
                            st.bar_chart(label_counts)

                        with col2:
                            st.write("**Distribusi Prediksi**")
                            fig_pie, ax_pie = plt.subplots()
                            ax_pie.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
                            ax_pie.axis('equal')
                            st.pyplot(fig_pie)
                        
                        # Visualisasi Pohon Keputusan
                        st.markdown("---")
                        st.subheader("🌳 Visualisasi Salah Satu Pohon Keputusan dari Model")
                        with st.spinner("Menggambar pohon keputusan..."):
                            fig_tree, ax_tree = plt.subplots(figsize=(25, 10))
                            plot_tree(
                                model.estimators_[0],  # Mengambil pohon pertama dari Random Forest
                                feature_names=vectorizer.get_feature_names_out(),
                                class_names=["FAKE", "REAL"],
                                filled=True,
                                max_depth=3,
                                fontsize=10,
                                ax=ax_tree
                            )
                            st.pyplot(fig_tree)

                        # Kesimpulan Dinamis (dipindahkan ke akhir)
                        st.markdown("---")
                        st.subheader("📝 Ringkasan Kesimpulan")
                        total_berita = len(df_uploaded)
                        jumlah_fake = label_counts.get('FAKE', 0)
                        jumlah_real = label_counts.get('REAL', 0)
                        persen_fake = (jumlah_fake / total_berita) * 100
                        persen_real = (jumlah_real / total_berita) * 100
                        mayoritas = "FAKE" if jumlah_fake > jumlah_real else "REAL"

                        if lang == "Indonesia":
                            kesimpulan_text = f"""
                            Dari total **{total_berita}** berita yang dianalisis:
                            - **{jumlah_fake} berita ({persen_fake:.1f}%)** terdeteksi sebagai **FAKE**.
                            - **{jumlah_real} berita ({persen_real:.1f}%)** terdeteksi sebagai **REAL**.
                            Secara keseluruhan, mayoritas berita dalam file Anda diklasifikasikan sebagai **{mayoritas}**.
                            """
                            st.write(kesimpulan_text)
                        else:
                            conclusion_text = f"""
                            Out of a total of **{total_berita}** news articles analyzed:
                            - **{jumlah_fake} articles ({persen_fake:.1f}%)** were detected as **FAKE**.
                            - **{jumlah_real} articles ({persen_real:.1f}%)** were detected as **REAL**.
                            Overall, the majority of the news in your file is classified as **{mayoritas}**.
                            """
                            st.write(conclusion_text)
                        
                        # Tombol Unduh
                        csv_output = df_uploaded.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Unduh Hasil Analisis" if lang == "Indonesia" else "Download Analysis Results",
                            data=csv_output,
                            file_name='hasil_klasifikasi_berita.csv',
                            mime='text/csv',
                        )
            else:
                st.error("❌ Peringatan: File CSV Anda harus memiliki kolom bernama 'title' dan 'text'.")
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
