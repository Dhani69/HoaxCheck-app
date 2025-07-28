# app.py
import streamlit as st
from predictor import classify_text

st.set_page_config(page_title="HoaxCheck", layout="wide")

# Welcome popup
if 'welcomed' not in st.session_state:
    st.session_state['welcomed'] = True
    st.info("COBA COBA COBA tes test tes**Peringatan**: Web ini adalah bagian dari **skripsi** dan hasil klasifikasi tidak bisa dipercaya 100%. Mohon untuk memverifikasi kembali berita yang diklasifikasikan.\n\nAplikasi ini bertujuan sebagai pembelajaran dan penelitian.")

# Exit button
if 'exit' not in st.session_state:
    st.session_state['exit'] = False
if st.sidebar.button("Exit"):
    st.session_state.clear()
    st.session_state['exit'] = True
if st.session_state.get('exit', False):
    st.title("Terima Kasih Sudah Menggunakan Website Ini")
    st.stop()

# Init flags
if 'classified' not in st.session_state:
    st.session_state['classified'] = False
if 'clear_inputs' not in st.session_state:
    st.session_state['clear_inputs'] = False

# Reset inputs if needed
if st.session_state['clear_inputs']:
    st.session_state['judul'] = ''
    st.session_state['isi'] = ''
    st.session_state['classified'] = False
    st.session_state['clear_inputs'] = False

# Title
st.title("📰 Klasifikasi Berita Hoax dengan XGBoost/LightGBM")

# Sidebar
with st.sidebar:
    st.header("Proses Klasifikasi Berita")
    st.markdown("""
    - **Preprocessing Teks**
    - **TF-IDF**
    - **Model Klasifikasi**: XGBoost atau LightGBM
    """)
    threshold = 0.5

# Input fields
judul = st.text_input("Masukkan Judul Berita", value=st.session_state.get('judul', ''), key='judul')
isi = st.text_area("Masukkan Isi Berita", value=st.session_state.get('isi', ''), key='isi')

# Model option
model_option = st.selectbox("Pilih Model Klasifikasi:", ["XGBoost", "LightGBM"])

# Classify button
if st.button("Klasifikasikan"):
    if judul.strip() == "" and isi.strip() == "":
        st.warning("Silakan isi minimal judul *atau* isi berita.")
    else:
        hasil, score, cleaned = classify_text(judul, isi, model_option)
        color = "red" if hasil == "HOAX" else "green"
        st.markdown(f'<h3 style="color:{color};">Hasil Klasifikasi: **{hasil}**</h3>', unsafe_allow_html=True)
       # st.write(f"**{hasil}** dengan skor probabilitas {score:.4f}")

        st.info("**Peringatan**: Web ini adalah bagian dari **skripsi** dan hasil klasifikasi tidak bisa dipercaya 100%. Mohon untuk memverifikasi kembali berita yang diklasifikasikan.\n\nAplikasi ini bertujuan sebagai pembelajaran dan penelitian.")

        with st.sidebar:
            st.subheader("Hasil Preprocessing")
            st.write(f"Teks yang dibersihkan: `{cleaned}`")
            st.subheader("Hasil Klasifikasi")
           # st.write(f"**{hasil}** dengan skor probabilitas {score:.4f}")

        st.session_state['classified'] = True

# Show ➕ button only after classified
if st.session_state.get('classified', False):
    if st.button("➕ Klasifikasi Berita Baru"):
        st.session_state['clear_inputs'] = True
        st.rerun()   # force rerun so inputs clear immediately
