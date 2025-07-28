# predictor.py
import joblib
import xgboost as xgb
from utils.preprocessing import clean_text

# Load models & vectorizer (lakukan sekali di awal)
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model2.json")
lgbm_model = joblib.load("models/lgbm_model2.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

threshold = 0.5

def classify_text(judul, isi, model_option):
    """
    Fungsi klasifikasi berita hoax.
    - judul: str
    - isi: str
    - model_option: "XGBoost" atau "LightGBM"
    Return: hasil ("HOAX"/"VALID"), score (float), cleaned_text (str)
    """
    # Gabungkan input
    if judul.strip() and isi.strip():
        combined_text = f"{judul} {isi}"
    elif judul.strip():
        combined_text = judul
        st.info("⚠️ Hanya Judul yang diinput. Hasil klasifikasi mungkin kurang akurat.\n\nUntuk memaksimalkan kerja aplikasi, dianjurkan untuk mengisi **Judul** dan **Isi Berita**")
    else:
        combined_text = isi
        st.info("⚠️ Hanya Isi yang diinput. Hasil klasifikasi mungkin kurang akurat.\n\nUntuk memaksimalkan kerja aplikasi, dianjurkan untuk mengisi **Judul** dan **Isi Berita**")

    # Preprocessing
    cleaned = clean_text(combined_text)
    vectorized = tfidf_vectorizer.transform([cleaned])

    # Prediksi
    if model_option == "XGBoost":
        dmatrix = xgb.DMatrix(vectorized, enable_categorical=False)
        pred_prob = xgb_model.predict(dmatrix, validate_features=False)
        score = pred_prob[0]
    else:
        pred_prob = lgbm_model.predict(vectorized, num_iteration=lgbm_model.best_iteration)
        score = pred_prob[0]

    pred_label = 1 if score > threshold else 0
    hasil = "HOAX" if pred_label == 1 else "VALID"

    return hasil, score, cleaned
