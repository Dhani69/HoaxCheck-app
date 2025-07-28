import streamlit as st
import joblib
import xgboost as xgb
from utils.preprocessing import clean_text  # Import fungsi preprocessing

# Load model dan vectorizer
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model2.json")  # Gunakan .json untuk XGBoost
lgbm_model = joblib.load("models/lgbm_model2.pkl")  # Gunakan .pkl untuk LightGBM
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Judul Aplikasi
st.title("📰 Klasifikasi Berita Hoax dengan XGBoost & LightGBM")

# Input dari pengguna
judul = st.text_input("Masukkan Judul Berita")
isi = st.text_area("Masukkan Isi Berita")

# Pilihan model
model_option = st.selectbox("Pilih Model Klasifikasi:", ["XGBoost", "LightGBM"])

# Threshold kustom (default 0.5)
threshold = st.slider("Threshold Deteksi HOAX", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if st.button("Klasifikasikan"):
    if judul.strip() == "" and isi.strip() == "":
        st.warning("Silakan isi minimal judul *atau* isi berita.")
    else:
        # Gabungkan input yang tersedia
        if judul.strip() and isi.strip():
            combined_text = f"{judul} {isi}"
        elif judul.strip():
            combined_text = judul
            st.info("⚠️ Hanya judul yang diinput. Hasil klasifikasi mungkin kurang akurat.")
        else:
            combined_text = isi
            st.info("⚠️ Hanya isi berita yang diinput. Hasil klasifikasi mungkin kurang akurat.")

        # Preprocessing
        cleaned = clean_text(combined_text)
        vectorized = tfidf_vectorizer.transform([cleaned])

        # Prediksi untuk XGBoost
        if model_option == "XGBoost":
            dmatrix_input = xgb.DMatrix(vectorized, enable_categorical=False)
            pred_prob = xgb_model.predict(dmatrix_input, validate_features=False)
            pred_label = [1 if prob > threshold else 0 for prob in pred_prob]
            score = pred_prob[0]  # Skor probabilitas untuk XGBoost

        # Prediksi untuk LightGBM
        else:
            pred_prob = lgbm_model.predict(vectorized, num_iteration=lgbm_model.best_iteration)
            pred_label = [1 if pred_prob > threshold else 0]
            score = pred_prob  # Skor probabilitas untuk LightGBM

        # Hasil klasifikasi
        hasil = "HOAX" if pred_label[0] == 1 else "VALID"
        st.success(f"Hasil Klasifikasi: **{hasil}**")
        st.write(f"🧪 Skor Probabilitas HOAX: `{score[0]:.4f}` (threshold: {threshold})")
















import streamlit as st
import joblib
import xgboost as xgb
from utils.preprocessing import clean_text  # Import fungsi preprocessing

# Load model dan vectorizer
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model2.json")  # Gunakan .json untuk XGBoost
lgbm_model = joblib.load("models/lgbm_model2.pkl")  # Gunakan .pkl untuk LightGBM
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Menyesuaikan tampilan untuk sidebar di kanan
st.set_page_config(page_title="Klasifikasi Berita Hoax", layout="wide")

# Judul Aplikasi
st.title("📰 Klasifikasi Berita Hoax dengan XGBoost & LightGBM")

# Sidebar untuk proses klasifikasi
with st.sidebar:
    st.header("Proses Klasifikasi Berita")
    
    # Pengaturan Threshold
    threshold = st.slider("Threshold Deteksi HOAX", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Input dari pengguna
judul = st.text_input("Masukkan Judul Berita")
isi = st.text_area("Masukkan Isi Berita")

# Pilihan model
model_option = st.selectbox("Pilih Model Klasifikasi:", ["XGBoost", "LightGBM"])

if st.button("Klasifikasikan"):
    if judul.strip() == "" and isi.strip() == "":
        st.warning("Silakan isi minimal judul *atau* isi berita.")
    else:
        # Gabungkan input yang tersedia
        if judul.strip() and isi.strip():
            combined_text = f"{judul} {isi}"
        elif judul.strip():
            combined_text = judul
            st.info("⚠️ Hanya judul yang diinput. Hasil klasifikasi mungkin kurang akurat.")
        else:
            combined_text = isi
            st.info("⚠️ Hanya isi berita yang diinput. Hasil klasifikasi mungkin kurang akurat.")

        # Preprocessing
        cleaned = clean_text(combined_text)
        vectorized = tfidf_vectorizer.transform([cleaned])

        # Sidebar: Menampilkan informasi tentang Preprocessing dan TF-IDF
        with st.sidebar:
            st.subheader("Hasil Preprocessing")
            st.write(f"Teks yang dibersihkan: `{cleaned}`")

        # Prediksi untuk XGBoost
        if model_option == "XGBoost":
            dmatrix_input = xgb.DMatrix(vectorized, enable_categorical=False)
            pred_prob = xgb_model.predict(dmatrix_input, validate_features=False)
            pred_label = [1 if prob > threshold else 0 for prob in pred_prob]
            score = pred_prob[0]  # Skor probabilitas untuk XGBoost

        # Prediksi untuk LightGBM
        else:
            pred_prob = lgbm_model.predict(vectorized, num_iteration=lgbm_model.best_iteration)
            pred_label = [1 if pred_prob > threshold else 0]
            score = pred_prob  # Skor probabilitas untuk LightGBM
            score = score[0]  # Pastikan mengambil nilai pertama dari array

        # Hasil klasifikasi
        hasil = "HOAX" if pred_label[0] == 1 else "VALID"
        st.success(f"Hasil Klasifikasi: **{hasil}**")

        # Sidebar: Menampilkan hasil klasifikasi dan threshold
        with st.sidebar:
            st.subheader("Hasil Klasifikasi")
            st.write(f"**{hasil}** dengan skor probabilitas {score:.4f}")
            st.write(f"Threshold yang digunakan: {threshold}")











import streamlit as st
import joblib
import xgboost as xgb
from utils.preprocessing import clean_text  # Import fungsi preprocessing

# Load model dan vectorizer
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model2.json")  # Gunakan .json untuk XGBoost
lgbm_model = joblib.load("models/lgbm_model2.pkl")  # Gunakan .pkl untuk LightGBM
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Menyesuaikan tampilan untuk sidebar di kanan
st.set_page_config(page_title="Klasifikasi Berita Hoax", layout="wide")

# Menampilkan Pop-up Selamat Datang Seperti Notifikasi
if 'welcomed' not in st.session_state:
    st.session_state['welcomed'] = True
    st.info("**Peringatan**: Web ini adalah bagian dari **skripsi** dan hasil klasifikasi tidak bisa dipercaya 100%. Mohon untuk memverifikasi kembali berita yang diklasifikasikan.\n\nAplikasi ini bertujuan sebagai pembelajaran dan penelitian.")

# Judul Aplikasi
st.title("📰 Klasifikasi Berita Hoax dengan XGBoost & LightGBM")

# Sidebar untuk proses klasifikasi
with st.sidebar:
    st.header("Proses Klasifikasi Berita")
    st.markdown("""
    **Proses klasifikasi berita hoax** terdiri dari beberapa langkah berikut:
    - **Preprocessing Teks**: Membersihkan teks dari karakter-karakter yang tidak penting.
    - **TF-IDF**: Mengubah teks yang sudah dibersihkan menjadi fitur numerik.
    - **Model Klasifikasi**: Menggunakan model **XGBoost** atau **LightGBM** untuk prediksi.
    - **Threshold**: Mengatur batas probabilitas untuk klasifikasi hoax.
    """)
    
    # Pengaturan Threshold
    threshold = st.slider("Threshold Deteksi HOAX", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Input dari pengguna
judul = st.text_input("Masukkan Judul Berita")
isi = st.text_area("Masukkan Isi Berita")

# Pilihan model
model_option = st.selectbox("Pilih Model Klasifikasi:", ["XGBoost", "LightGBM"])

if st.button("Klasifikasikan"):
    if judul.strip() == "" and isi.strip() == "":
        st.warning("Silakan isi minimal judul *atau* isi berita.")
    else:
        # Gabungkan input yang tersedia
        if judul.strip() and isi.strip():
            combined_text = f"{judul} {isi}"
        elif judul.strip():
            combined_text = judul
            st.info("⚠️ Hanya judul yang diinput. Hasil klasifikasi mungkin kurang akurat.")
        else:
            combined_text = isi
            st.info("⚠️ Hanya isi berita yang diinput. Hasil klasifikasi mungkin kurang akurat.")

        # Preprocessing
        cleaned = clean_text(combined_text)
        vectorized = tfidf_vectorizer.transform([cleaned])

        # Sidebar: Menampilkan informasi tentang Preprocessing dan TF-IDF
        with st.sidebar:
            st.subheader("Hasil Preprocessing")
            st.write(f"Teks yang dibersihkan: `{cleaned}`")
            st.write(f"Jumlah Fitur TF-IDF: {vectorized.shape[1]}")

        # Prediksi untuk XGBoost
        if model_option == "XGBoost":
            dmatrix_input = xgb.DMatrix(vectorized, enable_categorical=False)
            pred_prob = xgb_model.predict(dmatrix_input, validate_features=False)
            pred_label = [1 if prob > threshold else 0 for prob in pred_prob]
            score = pred_prob[0]  # Skor probabilitas untuk XGBoost

        # Prediksi untuk LightGBM
        else:
            pred_prob = lgbm_model.predict(vectorized, num_iteration=lgbm_model.best_iteration)
            pred_label = [1 if pred_prob > threshold else 0]
            score = pred_prob  # Skor probabilitas untuk LightGBM
            score = score[0]  # Pastikan mengambil nilai pertama dari array

        # Hasil klasifikasi dengan warna berbeda
        hasil = "HOAX" if pred_label[0] == 1 else "VALID"
        
        # Mengubah warna hasil klasifikasi
        if hasil == "HOAX":
            st.markdown(f'<h3 style="color:red;">Hasil Klasifikasi: **{hasil}**</h3>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h3 style="color:green;">Hasil Klasifikasi: **{hasil}**</h3>', unsafe_allow_html=True)

        # Sidebar: Menampilkan hasil klasifikasi dan threshold
        with st.sidebar:
            st.subheader("Hasil Klasifikasi")
            st.write(f"**{hasil}** dengan skor probabilitas {score:.4f}")
            st.write(f"Threshold yang digunakan: {threshold}")


st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Picture_icon_BLACK.svg/800px-Picture_icon_BLACK.svg.png", caption="Proses Klasifikasi")
















import streamlit as st
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from utils.preprocessing import clean_text  # Import dari modul utils

# Load model dan vectorizer
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model.json")  # HANYA load dengan .json
lgbm_model = joblib.load("models/lgbm_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Judul Aplikasi
st.title("📰 Klasifikasi Berita Hoax dengan XGBoost & LightGBM")

# Input dari pengguna
judul = st.text_input("Masukkan Judul Berita")
isi = st.text_area("Masukkan Isi Berita")

# Pilihan model
model_option = st.selectbox("Pilih Model Klasifikasi:", ["XGBoost", "LightGBM"])

# Threshold kustom
threshold = st.slider("Threshold Deteksi HOAX", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if st.button("Klasifikasikan"):
    if judul.strip() == "" and isi.strip() == "":
        st.warning("Silakan isi minimal judul *atau* isi berita.")
    else:
        # Gabungkan input yang tersedia
        if judul.strip() and isi.strip():
            combined_text = f"{judul} {isi}"
        elif judul.strip():
            combined_text = judul
            st.info("⚠️ Hanya judul yang diinput. Hasil klasifikasi mungkin kurang akurat.")
        else:
            combined_text = isi
            st.info("⚠️ Hanya isi berita yang diinput. Hasil klasifikasi mungkin kurang akurat.")

        # Preprocessing
        cleaned = clean_text(combined_text)
        vectorized = tfidf_vectorizer.transform([cleaned])

        # Prediksi
        if model_option == "XGBoost":
            dmatrix_input = xgb.DMatrix(vectorized, enable_categorical=False)
            pred_prob = xgb_model.predict(dmatrix_input, validate_features=False)
            pred = [1 if prob > threshold else 0 for prob in pred_prob]
        else:
            pred_prob = lgbm_model.predict_proba(vectorized)[0][1]  # Probabilitas kelas 1 (hoax)
            pred = [1 if pred_prob > threshold else 0]

        hasil = "HOAX" if pred[0] == 1 else "BUKAN HOAX"
        st.success(f"Hasil Klasifikasi: **{hasil}**")
        st.write(f"🧪 Skor Probabilitas HOAX: `{pred_prob:.4f}` (threshold: {threshold})")

        # ROC Curve (untuk visualisasi dan evaluasi)
        fpr, tpr, thresholds = roc_curve(y_test, pred_prob)  # Prediksi pada X_test
        roc_auc = auc(fpr, tpr)

        # Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Tampilkan ROC curve di Streamlit
        st.pyplot(plt)







import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load dataset hasil TF-IDF
df_tfidf = pd.read_csv('D:\KULIAH\Skripsi\dataset berita\data final TFIDF.csv')  # Gantilah path sesuai dengan file Anda

# Pisahkan fitur dan label
X = df_tfidf.drop('label', axis=1)  # Pastikan kolom 'label' ada dalam dataset
y = df_tfidf['label']

# Inisialisasi SMOTE
smote = SMOTE(random_state=42)

# Terapkan SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)

# Gabungkan kembali ke DataFrame
df_smote = pd.DataFrame(X_resampled, columns=X.columns)
df_smote['label'] = y_resampled

# Cek distribusi label
label_counts = df_smote['label'].value_counts()

print("Distribusi Label Setelah SMOTE:")
print(label_counts)

# Persentase distribusi label
label_percentage = df_smote['label'].value_counts(normalize=True) * 100

print("\nPersentase Distribusi Label Setelah SMOTE:")
print(label_percentage)

# Simpan dataset hasil SMOTE
df_smote.to_csv('D:\KULIAH\Skripsi\dataset berita\data_final_SMOTE.csv', index=False)

print("✅ SMOTE berhasil diterapkan! Dataset tersimpan.")
