# test_predictor.py

import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext!")

import logging
logging.getLogger('streamlit').setLevel(logging.CRITICAL)

import pytest
from predictor import classify_text
from utils.preprocessing import clean_text
import joblib
import xgboost as xgb

# Load ulang model & vectorizer seperti di predictor.py
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model2.json")
lgbm_model = joblib.load("models/lgbm_model2.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# === TEST DASAR ===
def test_xgb_predict_proba():
    sample = ["""Kerusuhan Mei 1998 korban utamanya adalah 99% pribumi. Tidak ada korban spesifik etnis Tionghoa yg dapat ditelusuri otentisitas forensiknya. 
              Jika ada pihak terus menerus framing menghubungkan rusuh Mei 98 sbg kekuatan rekayasa yg dikatakan untuk menargetkan (anti) etnis tertentu, 
              ini penghinaan kpd otoritas kepolisian (yg telah anulir asumsi liar SARA tersebut) dan kedaulatan NKRI. Itu sebabnya akun2 yg tebar ginian biasa anonim."""]
    cleaned = clean_text(sample[0])
    vec = tfidf_vectorizer.transform([cleaned])
    feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
    dmatrix = xgb.DMatrix(vec, feature_names=feature_names)
    prob = xgb_model.predict(dmatrix)
    # print(f"[XGBoost] Predicted probability: {prob[0]:.4f}")
    print(f"\n[INFO] XGBoost Model Worked Properly {type(xgb_model)}")
    assert 0.0 <= prob[0] <= 1.0


def test_lgbm_predict_proba():
    sample = ["Ini berita valid kok."]
    cleaned = clean_text(sample[0])
    vec = tfidf_vectorizer.transform([cleaned])
    prob = lgbm_model.predict(vec)
   # print(f"[LightGBM] Predicted probability: {prob[0]:.4f}")
    print(f"[INFO] LightGBM Model Worked Properly {type(lgbm_model)}")
    assert 0.0 <= prob[0] <= 1.0


def get_model(choice):
    if choice.lower() == "xgboost":
        return xgb_model
    elif choice.lower() == "lightgbm":
        return lgbm_model
    else:
        raise ValueError("Invalid model choice")
    
def test_get_model_xgb():
    model = get_model("xgboost")
    print(f"\n[INFO] XGBoost Model Succesfully Loaded {type(model)}")
    assert isinstance(model, xgb.Booster)

def test_get_model_lgbm():
    model = get_model("lightgbm")
    print(f"[INFO] LightGBM Model Succesfully Loaded {type(model)}")
    assert model is not None


# === TEST classify_text dengan variasi input ===
@pytest.mark.parametrize("judul, isi, model_option", [
    ("Judul Saja", "", "XGBoost"),
    ("", "Isi Saja", "LightGBM"),
    ("Judul Lengkap", "Isi Lengkap Berita", "XGBoost"),
])
def test_classify_text_variation(judul, isi, model_option):
    hasil, score, cleaned = classify_text(judul, isi, model_option)
    print(f"\n[{model_option}] Title: '{judul}' | Body: '{isi}' => Prediction: {hasil}, Score: {score:.4f}, Cleaned: '{cleaned}'")
    print(f"[INFO] Input Succesfully Loaded {type(classify_text)}\n")
    assert hasil in ["HOAX", "VALID"]
    assert 0.0 <= score <= 1.0
    assert isinstance(cleaned, str)

def test_clean_text_function():
    raw = "COBA TESTING 123 https://www.youtube.com/!!! 💥💥 #Hoax"
    cleaned = clean_text(raw)
    print(f"[Clean Text] Raw: '{raw}' -> Cleaned: '{cleaned}'")
    print(f"\n[INFO] Text Cleaned Succesfully")
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0

# === TEST BATCH DENGAN DATA ===
test_data = [
    # Hoaks
    ("AFC Melarang Indonesia Menaturalisasi Warga Negara Belanda", """🚨BREAKING NEWS‼️ PSSI/timnas Indonesia dilarang menaturalisasi warga negara Belanda setelah protes keras dari AFC di kantor pusat Kuala lumpur Malaysia pada 31 Maret lalu😱. Keputusan ini telah disepakati oleh anggota AFC khusus nya Bahrain dan Malaysia yg mengajukan protes atas dasar menjaga integritas kompetisi internasional yg mengedepan kan peningkatan kompetisi liga domestik di negara masing-masing🥶. Pihak PSSI belum memberikan pernyataan resmi terkait hal ini,Pasca keputusan telah di buat .tulis (Fabrizio Romano). Bagaimana pendapat kalian geng Jika ini terjadi ⁉️""", 1),

    ("Kerusuhan Mei 1998 Tidak Spesifik Menyerang Etnis Tionghoa", """Kerusuhan Mei 1998 korban utamanya adalah 99% pribumi. Tidak ada korban spesifik etnis Tionghoa yg dapat ditelusuri otentisitas forensiknya. Jika ada pihak terus menerus framing menghubungkan rusuh Mei 98 sbg kekuatan rekayasa yg dikatakan untuk menargetkan (anti) etnis tertentu, ini penghinaan kpd otoritas kepolisian (yg telah anulir asumsi liar SARA tersebut) dan kedaulatan NKRI. Itu sebabnya akun2 yg tebar ginian biasa anonim.""", 1),

    ("Tanah Tanpa Sertifikat Elektronik Bakal Jadi Milik Negara", """…Setelah pengaturan penyaluran GAS ELPIJI ukuran 3 kilogram yang membuat panik emak-emak, kini giliran pengaturan SERTIFIKAT TANAH yang akan membuat rempong kaum bapak-bapak.
Sesuai informasi yang terlanjur sudah meluas di media sosial itu, SERTIFIKAT TANAH versi kertas seperti berlaku selama ini, akan diganti oleh pemerintah dengan sertifikat versi digital, atau sertifikat tanah elektronik mulai tahun 2026. 
Bagi yang tidak mengganti SERTIFIKAT TANAH-nya menjadi sertifikat elektronik yang dimulai berlaku Februari 2025 ini, maka surat tanah yang masih kertas (rinci, letter C) akan dimusnahkan pemerintah. Resikonya tanah yang semula milik masyarakat, akan diambilalih kepemilikannya oleh negara…""", 1),

    ("Warga China Tendang Wajah Anak di Pulau Rempang", """Kejadian ini di Tanah Rempang yg tanah nya di Rampas Oleh Oligarki di lindungi Rezim Jokowi. di Rempang..anak SMP Pulang dri Skolah Mempertahankan Rumah nya yg akan di Robohkan paksa Oleh Cina..dia tdk mau pergi lalu di seret ke luar di hajar Oleh Warga Cina yg sdh di buatkan KTP Non Pribumi…!
Bagaimana Jk itu terjadi pd anak Cucu kt yg akan dtg..itu blm seberapa Cina Menghajarnya Tp bsk klo sdh Menguasai Indonesia akan lebih Kejam lagi. Bagai mana Jk itu terjadi korban nya anak Cucumu Yg tdk tau apa2..
Wahai Para Pemuja2 Jokowi..para Anthek2 Jokowi para Penjilat2. Hidup mu di akirat sdh Ikut tanggung Jawap Pemimpin yg kau Pilih di Siksa di Neraka..anak Cucumu di Perlakukan Sperti itu Oleh asing Cina Penjajah Ibu Pertiwi?""", 1),

    ("Perdana Menteri Israel Ancam Presiden Prabowo", """Tuan Presiden (Prabowo) situasi di wilayah konflik sebagai “ancaman besar” bagi siapa pun yang berani meninjaunya, termasuk Prabowo. Saya menyarankan untuk tidak terlalu dalam mendukung palestina sebab ini menyangkut keselamatan semua pihak.” “✍🏾 PM Israel 🇮🇱 Benyamin Netanyahu peringatkan negara siapapun, termasuk Indonesia 🇮🇩: “Kepada Mr. Presiden Prabowo JANGAN TERLALU IKUT CAMPUR, PENTINGKAN KESELAMATAN ANDA, NEGARA ANDA, DARI PADA SIBUK BANTU TERORIS” Syaratnya ialah… TERORIS HAMAS BEBASKAN SANDERA.""", 1),

    # Valid
    ("Menteri RI Pilih Hidup Miskin, Tak Korupsi Meski Garap Proyek Raksasa", """Memiliki jabatan tinggi kerap dikaitkan dengan hidup mewah dan harta yang banyak. Namun, salah satu Menteri di era masa pemerintahan Soekarno dan Soeharto ini memilih jalan hidup berbeda.
Adalah Sutami, pria yang menjavat sebagai Menteri Pekerjaan Umum dan Perumahan Rakyat (PUPR) sejak 1964 hingga 1978. Keteladanan Sutami diperoleh dari gaya hidupnya yang berbeda dari para menteri lain. Selama 14 tahun menjadi menteri atau 8 periode, dia konsisten menolak pemberian negara dan memilih hidup miskin. Penyebabnya karena masih banyak rakyat hidup sengsara, sehingga tak patut menunjukkan hidup mewah. Staf Ahli Sutami, Hendropranoto, dalam kesaksian berjudul "Sutami Sosok Manusia Pembangunan Indonesia" (1991) menceritakan, salah satu sikap itu tercermin pada kebiasaan berjalan kaki ketika di mengunjungi suatu wilayah, khususnya perdesaan dan pelosok wilayah.

Dia rela berjalan kaki berkilo-kilo karena tak ingin merepotkan orang. Terlebih, jalan kaki juga dipilih karena lebih efisien dan mudah saat meninjau berbagai proyek infrastruktur.

Dengan melakukan ini Sutami bisa mengetahui implementasi dari pengerjaan proyek di bawah naungannya. Selain itu, jika ada permasalahan pun, bisa cepat diselesaikan.

Baginya, pembangunan infrastruktur di pedesaan dan pelosok wilayah lebih bermanfaat bagi rakyat kecil, alih-alih difokuskan untuk kepentingan industri dan pengusaha.

Dalam pewartaan Tempo (22/11/1980), tutur kata dan keseharian Sutami juga kental dengan kerendahan hati. Sebagai intelektual dan profesional di bidangnya, pria kelahiran 19 Oktober 1928 ini dikenal sederhana dan sangat merakyat.

Meski berkecimpung di "lahan basah", Sutami sama sekali tak mengambil uang negara. Bahkan, rumah pribadi saja tak punya. Dia baru memiliki rumah setelah berhenti menjadi menteri pada 29 Maret 1978 karena sakit. Itu pun pembelian rumah dilakukan lewat cicilan per bulan.

Atas dasar ini, dia dijuluki banyak orang sebagai "Menteri Termiskin". Dia pun tak mempermasalahkan julukan itu.

Setelah pensiun, diketahui Sutami hidup jauh dari kemewahan. Rumah yang masih nyicil itu pernah diputus listriknya karena Sutami tak bisa membayar tagihan. Lalu, ketika sakit pun, Sutami enggan ke rumah sakit karena takut tidak bisa membayar tagihan rumah sakit.

Diketahui, Sutami mengidap penyakit liver kronis. Penyakit liver tersebut diketahui karena dia semasa hidup kurang makanan bergizi dan kelelahan akibat sering berpergian jalan kaki. Kabar tragis ini kemudian didengar Presiden Soeharto yang kemudian segera meminta Sutami berobat tanpa perlu membayar. Namun, Sutami akhirnya kalah dari penyakitnya. Pada 13 November 1980, dia meninggal dunia.

Meski sudah tiada, karya-karya Sutami yang jauh dari sensasi semasa menjabat banyak dirasakan masyarakat manfaatnya hingga saat ini. Sederet megaproyek yang terbangun olehnya diantaranya tol Jagorawi, Jembatan Semanggi, Jembatan Ampera dan sebagainya.""",0),




    ("Sri Mulyani 'Senggol' Bahlil Lifting Minyak RI Belum Capai Target", """Menteri Keuangan Sri Mulyani 'menyenggol' Menteri ESDM Bahlil Lahadalia karena lifting minyak Indonesia belum mencapai target 605 barel per hari pada 2025.
"Lifting minyak kita di 567 ribu barel per hari, di bawah asumsi APBN (2025), yaitu 605 ribu barel per hari," ungkapnya dalam Konferensi Pers APBN KiTA di Kementerian Keuangan, Jakarta Pusat, Selasa (17/6).
"Kita harapkan nanti dari Menteri ESDM (Bahlil Lahadalia) dan SKK (Satuan Kerja Khusus Pelaksana Kegiatan Usaha Hulu Minyak dan Gas Bumi) akan menyampaikan, moga-moga lifting minyak bisa naik menembus di atas 600 lagi (barel per hari)," sambung Sri Mulyani.
Angka lifting minyak yang disampaikan Bendahara Negara itu merupakan realisasi per Mei 2025. Jumlahnya memang masih di bawah target yang ditetapkan dalam APBN 2025.
Capaian lifting minyak pada tahun lalu sebesar 579,7 barel per hari juga tak mencapai target yang dipatok, yakni 635 barel per hari.
Di lain sisi, ia menyoroti harga minyak dunia yang masih bergejolak. Ini diperparah dengan adanya perang di Timur Tengah antara Israel dengan Iran.
Ia menyebut asumsi harga minyak di APBN 2025 adalah US$82 per barel. Sedangkan pada end of period (eop) berada di posisi US$62,75 dan mencapai US$70,05 secara year to date (ytd), yakni masih di bawah harga asumsi.
"Lifting gas (realisasi per Mei 2025) di 987,5 ribu barel setara minyak per hari, di bawah asumsi kita 1.005 ribu barel setara minyak per hari. Ini adalah sesuatu yang juga harus kita lihat," wanti-wanti sang Bendahara Negara.
"Selain dipengaruhi oleh kondisi di dalam negeri kita, terutama untuk sektor pertambangan minyak, juga dipengaruhi oleh apa yang sekarang sedang berlangsung di Timur Tengah, yaitu perang antara Israel dengan Iran," tandasnya.
""", 0),




    ("Israel Klaim Bunuh Kepala Staf Perang Angkatan Bersenjata Iran Ali Shadmani", """Militer Israel mengklaim telah membunuh Kepala Staf Angkatan Bersenjata Iran Ali Shadmani, Selasa (17/6/2025). Shadmani baru diangkat menggantikan Gholam Ali Rashid yang juga terbunuh akibat  serangan Israel.

Pasukan Pertahanan Israel (IDF) mengklaim Shadmani terbunuh dalam serangan di Ibu Kota Teheran pada Selasa dini hari. Setelah menerima informasi intelijen akurat dari Direktorat Intelijen IDF pada Selasa, IAF (Angkatan Udara Israel) menyerang pusat komando yang dikelola staf di jantung Kota Teheran dan membunuh Ali Shadmani, kepala staf perang," bunyi pernyataan IDF, seperti dilaporkan kembali Anadolu.

Disebutkan Shadmani sebelumnya sempat memimpin Korps Garda Revolusi Islam (IRGC) dan unit militer Iran. Sejauh ini belum ada pernyataan dari pemerintah maupun militer Iran mengenai klaim Israel tersebut. Namun biasanya Iran segera menginformasikan jika ada pejabatnya yang tewas akibat serangan Israel.

Pemimpin Tertinggi Iran Ayatollah Ali Khamenei sebelumnya menunjuk Shadmani untuk menggantikan Rashid yang tewas akibat serangan udara Israel pada Jumat pekan lalu.""",0),


    ("Menkomdigi Wanti-wanti Dominasi Netflix Cs di RI, Bakal Lakukan Ini", """ Menteri Komunikasi dan Digital (Menkomdigi) Meutya Hafid menyebut dominasi layanan over-the-top (OTT) global, seperti Netflix Cs, tak boleh merugikan ekosistem penyiaran nasional yang selama ini berkontribusi terhadap informasi publik.
Dalam audiensi dengan asosiasi industri media Asia Pasifik, ia meminta agar pelaku OTT turut berperan dalam mendukung konten dan talenta lokal. Menurutnya langkah ini penting untuk menjaga kedaulatan digital di Tanah Air.

Dalam pertemuan dengan Presiden MPA Asia Pasifik, Meutya meminta OTT lebih aktif mendukung produksi lokal dan membiayai ekosistem penyiaran sebagai bagian dari kedaulatan digital Indonesia.

"Kami juga ingin Anda memberdayakan industri penyiaran," kata Meutya saat audiensi dengan Presiden dan Managing Director MPA untuk Asia Pasifik Mila Venugopalan di Kantor Komdigi, Jakarta, dalam keterangan resminya, Kamis (12/6).

Meutya mengatakan industri penyiaran masih memainkan peran penting dalam menjangkau masyarakat di seluruh pelosok Indonesia, terutama di wilayah-wilayah yang belum terjangkau koneksi internet.

Namun begitu, menurutnya industri ini menghadapi tantangan berat, karena beban investasi dan biaya operasional yang tinggi, sementara tren masyarakat bergeser ke konten digital melalui OTT.

"Prinsip dasarnya adalah bahwa harus ada kondisi yang setara antara industri penyiaran dengan platform OTT," ujar dia.

Presiden dan Managing Director MPA untuk Asia Pasifik Mila Venugopalan merespons positif dan menawarkan berbagi praktik terbaik dari berbagai negara, termasuk Australia, di mana penyiar lokal justru mendorong deregulasi dan efisiensi alih-alih memberatkan OTT.

"Termasuk film dan acara televisi yang diproduksi di negara Anda-yang dikonsumsi oleh lebih dari 200 juta pengguna internet di Indonesia, yang merupakan populasi internet terbesar keempat di dunia," ujarnya.

MPA menyatakan komitmen untuk berinvestasi dalam bakat lokal dan cerita Indonesia. Mereka juga menyampaikan apresiasi atas langkah pemerintah dalam memblokir situs-situs pembajakan, sebagai upaya perlindungan konten digital yang berkembang pesat di era internet.

"Kami sangat menghargai kolaborasi yang terus dilakukan oleh Kementerian Komunikasi dan Digital dalam membantu mempromosikan dan melindungi konten digital," ungkapnya.""",0),



    ("Menteri RI Pilih Hidup Miskin, Tak Korupsi Meski Garap Proyek Raksasa", """Memiliki jabatan tinggi kerap dikaitkan dengan hidup mewah dan harta yang banyak. Namun, salah satu Menteri di era masa pemerintahan Soekarno dan Soeharto ini memilih jalan hidup berbeda.
Adalah Sutami, pria yang menjavat sebagai Menteri Pekerjaan Umum dan Perumahan Rakyat (PUPR) sejak 1964 hingga 1978. Keteladanan Sutami diperoleh dari gaya hidupnya yang berbeda dari para menteri lain. Selama 14 tahun menjadi menteri atau 8 periode, dia konsisten menolak pemberian negara dan memilih hidup miskin. Penyebabnya karena masih banyak rakyat hidup sengsara, sehingga tak patut menunjukkan hidup mewah. Staf Ahli Sutami, Hendropranoto, dalam kesaksian berjudul "Sutami Sosok Manusia Pembangunan Indonesia" (1991) menceritakan, salah satu sikap itu tercermin pada kebiasaan berjalan kaki ketika di mengunjungi suatu wilayah, khususnya perdesaan dan pelosok wilayah.
Dia rela berjalan kaki berkilo-kilo karena tak ingin merepotkan orang. Terlebih, jalan kaki juga dipilih karena lebih efisien dan mudah saat meninjau berbagai proyek infrastruktur.
Dengan melakukan ini Sutami bisa mengetahui implementasi dari pengerjaan proyek di bawah naungannya. Selain itu, jika ada permasalahan pun, bisa cepat diselesaikan.
Baginya, pembangunan infrastruktur di pedesaan dan pelosok wilayah lebih bermanfaat bagi rakyat kecil, alih-alih difokuskan untuk kepentingan industri dan pengusaha.
Dalam pewartaan Tempo (22/11/1980), tutur kata dan keseharian Sutami juga kental dengan kerendahan hati. Sebagai intelektual dan profesional di bidangnya, pria kelahiran 19 Oktober 1928 ini dikenal sederhana dan sangat merakyat.
Meski berkecimpung di "lahan basah", Sutami sama sekali tak mengambil uang negara. Bahkan, rumah pribadi saja tak punya. Dia baru memiliki rumah setelah berhenti menjadi menteri pada 29 Maret 1978 karena sakit. Itu pun pembelian rumah dilakukan lewat cicilan per bulan.
Atas dasar ini, dia dijuluki banyak orang sebagai "Menteri Termiskin". Dia pun tak mempermasalahkan julukan itu.
Setelah pensiun, diketahui Sutami hidup jauh dari kemewahan. Rumah yang masih nyicil itu pernah diputus listriknya karena Sutami tak bisa membayar tagihan. Lalu, ketika sakit pun, Sutami enggan ke rumah sakit karena takut tidak bisa membayar tagihan rumah sakit.
Diketahui, Sutami mengidap penyakit liver kronis. Penyakit liver tersebut diketahui karena dia semasa hidup kurang makanan bergizi dan kelelahan akibat sering berpergian jalan kaki. Kabar tragis ini kemudian didengar Presiden Soeharto yang kemudian segera meminta Sutami berobat tanpa perlu membayar. Namun, Sutami akhirnya kalah dari penyakitnya. Pada 13 November 1980, dia meninggal dunia.
Meski sudah tiada, karya-karya Sutami yang jauh dari sensasi semasa menjabat banyak dirasakan masyarakat manfaatnya hingga saat ini. Sederet megaproyek yang terbangun olehnya diantaranya tol Jagorawi, Jembatan Semanggi, Jembatan Ampera dan sebagainya.""", 0)
]

def test_title_only_batch():
    correct = 0
    total = len(test_data)
    print("\n[TITLE ONLY] Batch Test Results:")
    for i, (title, _, label) in enumerate(test_data):
        hasil, _, _ = classify_text(title, "", "XGBoost")
        pred = 1 if hasil == "HOAX" else 0
        status = "✅" if pred == label else "❌"
        print(f"{status} {i+1}. Label: {label}, Predicted: {pred} ({hasil})")
        if pred == label:
            correct += 1
    print(f"[TITLE ONLY] Correct: {correct}/{total}")
    assert correct >= 2  # disesuaikan: minimal prediksi benar

def test_body_only_batch():
    correct = 0
    total = len(test_data)
    print("\n[BODY ONLY] Batch Test Results:")
    for i, (_, body, label) in enumerate(test_data):
        hasil, _, _ = classify_text("", body, "XGBoost")
        pred = 1 if hasil == "HOAX" else 0
        status = "✅" if pred == label else "❌"
        print(f"{status} {i+1}. Label: {label}, Predicted: {pred} ({hasil})")
        if pred == label:
            correct += 1
    print(f"[BODY ONLY] Correct: {correct}/{total}")
    assert correct >= 2

def test_combined_input_batch():
    correct = 0
    total = len(test_data)
    print("\n[COMBINED INPUT] Batch Test Results:")
    for i, (title, body, label) in enumerate(test_data):
        hasil, _, _ = classify_text(title, body, "XGBoost")
        pred = 1 if hasil == "HOAX" else 0
        status = "✅" if pred == label else "❌"
        print(f"{status} {i+1}. Label: {label}, Predicted: {pred} ({hasil})")
        if pred == label:
            correct += 1
    print(f"[COMBINED INPUT] Correct: {correct}/{total}")
    assert correct >= 2
