import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext!")

import logging
logging.getLogger('streamlit').setLevel(logging.CRITICAL)

import pytest
import joblib
import xgboost as xgb
from app import classify
from utils.preprocessing import clean_text



# Load models dan vectorizer dari file seperti di aplikasi Streamlit
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model2.json")
lgbm_model = joblib.load("models/lgbm_model2.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# === TEST AWAL ===
def test_model_predict_proba_xgb():
    sample = ["Peluang sudah di depan mata tapi lo masih ragu?..."]
    cleaned = clean_text(sample[0])
    vec = tfidf_vectorizer.transform([cleaned])
    feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
    dmatrix = xgb.DMatrix(vec, feature_names=feature_names)
    prob = xgb_model.predict(dmatrix)
    print(f"[XGBoost] Predicted probability: {prob[0]:.4f}")
    print(f"[INFO] XGBoost Model Worked Properly {type(xgb_model)}")
    assert 0.0 <= prob[0] <= 1.0

def test_model_predict_proba_lgbm():
    sample = ["berita ini valid"]
    cleaned = clean_text(sample[0])
    vec = tfidf_vectorizer.transform([cleaned])
    prob = lgbm_model.predict(vec)
    print(f"[LightGBM] Predicted probability: {prob[0]:.4f}")
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
    print(f"[INFO] XGBoost Model Succesfully Loaded {type(model)}")
    assert isinstance(model, xgb.Booster)

def test_get_model_lgbm():
    model = get_model("lightgbm")
    print(f"[INFO] LightGBM Model Succesfully Loaded {type(model)}")
    assert model is not None

@pytest.mark.parametrize("judul, isi", [
    ("Judul Saja", ""),
    ("", "Isi Saja"),
    ("Judul", "Isi Berita"),
])
def test_classify_input_variation(judul, isi):
    hasil, score = classify(judul, isi, model_option="XGBoost")
    print(f"[Input Variation] Title: '{judul}' | Body: '{isi}' => Prediction: {hasil}, Score: {score:.4f}")
    print(f"[INFO] Input Succesfully Loaded {type(classify)}")
    assert hasil in ["HOAX", "VALID"]
    assert 0.0 <= score <= 1.0

def test_clean_text_output():
    raw_text = "INI BERITA!!! 💥💥 HOAX BANGET deh..."
    cleaned = clean_text(raw_text)
    print(f"\n[Clean Text] Input: {raw_text}\nCleaned: {cleaned}")
    print(f"\n[INFO] Text Cleaned Succesfully")
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0

# === TEST KHUSUS 10 BERITA ===
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
    ("Kasino di Bandung Menyaru Tempat Futsal dan Karaoke", """New Ballroom Billiard, Karaoke, and Live Music di Jalan Ahmad Yani No.126, Kelurahan Malabar, Kecamatan Lengkong, Kota Bandung, Selasa dini hari, 17 Juni 2025, tiba-tiba ramai didatangi puluhan polisi.
Ada penggerebekan narkoba? Ternyata Kepolisian Daerah Jawa Barat sedang merazia tempat biliar dan karaoke itu karena menyediakan fasilitas untuk judi kasino. Tidak tanggung-tanggung, polisi menangkap 63 orang terdiri dari pemain dan bandarnya.
Tempat hiburan dan lapangan futsal itu, rupanya menyediakan peralatan kasino lengkap dengan kursi empuk berwarna hitam.
Kabid Humas Polda Jabar, Kombes Hendra Rochmawan mengatakan penggerebekan itu berdasarkan hasil patroli siber serta laporan masyarakat adanya praktik perjudian di lokasi tersebut.
"Sementara baru kita amankan dulu, kita lidik dulu setelah itu tentu akan dilakukan gelar perkara setelah itu pasal-pasal. Dan ada uang cash Rp359 juta yang sudah kita sita sebagai barang bukti," kata Hendra di Bandung, Selasa.
Hendra mengungkapkan tempat tersebut berkamuflase sebagai sarana olahraga futsal dan tempat hiburan. Namun di dalamnya, polisi menemukan 10 meja judi serta perlengkapan lain seperti dadu, koin pengganti uang, dan peralatan elektronik pendukung.
Menurut dia, tempat perjudian ini memiliki pembagian area, mulai dari meja standar dengan taruhan minimal Rp300 ribu hingga ruang VIP dengan taruhan mulai dari Rp3 juta.
“Di ruang VIP ini kita lihat meja bagus, ruangan eksklusif dan ini secara singkat kita sampaikan dari penyidik bahwa pemain di sini eksklusif, dengan taruhan minimal Rp3 juta sampai tidak ada terhitung,” katanya.
Dari lokasi tersebut, polisi menyita sejumlah barang bukti di antaranya, uang tunai sebesar Rp359 juta, 10 set meja kasino untuk permainan jenis Niu Niu dan Baccarat, empat buku rekening bank, 38 unit telepon genggam, satu unit iPad, satu perangkat komputer kasir dan sejumlah kamera CCTV dan perangkat monitor.
“Lama operasionalnya masih kita lakukan penyelidikan. Apakah sudah lama atau masih baru, dan ini catatan kita untuk penyelidikan dan penyidikan,” kata Hendra.
Ia menambahkan, pihaknya juga masih menyelidiki legalitas tempat tersebut, termasuk izin operasional, peredaran minuman keras, dan kemungkinan adanya unsur tindak pidana lain.
“Tentu, semua yang ada di sini yang berkaitan dengan legalitas. Termasuk tadi kita lakukan pemeriksaan urine apakah mereka menggunakan narkoba, sudah kita lakukan pemeriksaan itu,” kata dia.
Lokasi judi kasino ini berada di pusat kota Bandung. Hendra menjelaskan penggerebekan dilakukan tim gabungan dari Subdit Siber Polda Jabar dan Satreskrim Polrestabes Bandung.
Adapun 63 orang yang ditangkap terdiri atas 37 karyawan, 23 pemain judi, dan tiga orang yang diduga sebagai penanggung jawab.
“Memang ini kondisinya sangat tersamar di keramaian kota, dan promosinya futsal. Kita lihat juga tadi ada barang bukti kendaraan yang dipakai pengguna, juga karyawan di sini dan posisinya tetap di sana,” kata dia.
“Di ruang VIP ini kita lihat table bagus, ruangan eksklusif dan ini secara singkat kita sampaikan dari penyidik bahwa pemain di sini eksklusif, dengan taruhan minimal Rp3 juta sampai tidak terhitung,” katanya.
Wacana Pelegalan Kasino
Penggerebekan ini terjadi di tengah munculnya wacana pelegalan kasino demi mendapat pemasukan dari pajak dan mencegah larinya uang besar dari warga Indonesia yang main judi di negara tetangga seperti Singapura, Malaysia atau Kamboja.
Pengamat Ekonomi Benny Batara yang akrab disapa Bennix mengatakan, kasino jauh berbeda dibandingkan praktik judi daring ilegal yang justru menyasar masyarakat kalangan bawah.
"Kalau kita melegalkan kasino, itu beda dengan judi online seperti yang marak di Kamboja. Judi online bisa diakses siapa pun dengan handphone, tukang ojek, tukang sayur, semua bisa ikut. Tapi kasino itu fisik. Harus beli tiket pesawat, sewa kamar hotel. Artinya, segmen pasarnya jelas, kalangan menengah ke atas," katanya kepada Antara, Kamis, 12 Juni 2025.
Ia mengatakan, legalisasi kasino secara strategis bisa memberi pemasukan besar ke negara dan mengalihkan aliran uang yang selama ini bocor ke luar negeri.
"Kalau judi itu legal, duit masuk ke kas negara lewat Direktorat Jenderal Pajak. Kalau ilegal, duit masuk ke oknum aparat. Pilihannya, kita mau memperkaya siapa hari ini," katanya.
Singapura yang hanya berpenduduk 6 juta orang namun berhasil mencetak pendapatan hingga Rp109 triliun dari dua kasino ternama, Marina Bay Sands dan Resorts World Sentosa.
Jumlah ini ditargetkan meningkat hingga Rp150 triliun tahun ini dengan target mayoritas pengunjung wisatawan negara tetangga, termasuk Indonesia.
"Warga Singapura sendiri dipersulit untuk berjudi. Mereka harus bayar tiket masuk kasino sekitar Rp35 juta. Tapi bagi warga asing, itu tidak berlaku. Karena memang kasino dibangun bukan untuk rakyat mereka. Mereka membidik orang Malaysia, Indonesia terutama dari Medan. Banyak pengusaha kita tiap akhir pekan sewa pesawat untuk ke sana," katanya.
Ia juga menyoroti kalangan atas berjudi bukan untuk menjadi kaya tetapi sebagai bentuk hiburan berisiko tinggi, berbeda dengan motif masyarakat bawah yang berjudi karena ingin cepat kaya.
"Mereka tahu mereka bisa rugi miliaran dan mereka datang dengan target kerugian itu. Tapi itu hiburan buat mereka. Mereka tidak mengopi di pinggir jalan. Mereka cari sensasi yang beda," ucapnya.
Seribu Triliun Rupiah Terbang
Menurut dia, selama Indonesia tidak mampu menyediakan sarana hiburan semacam itu, maka uang akan terus mengalir ke luar negeri. Bahkan ia menyebut Indonesia kehilangan potensi ratusan triliun rupiah tiap tahun karena tidak mengelola potensi industri kasino secara sah.
"Selama 10 tahun ini, sudah lebih dari 1.000 triliun rupiah uang orang Indonesia terbang ke luar negeri buat judi," ucapnya.
Bennix juga mengungkap potensi campur tangan asing dalam menggagalkan potensi ekonomi nasional, termasuk investasi strategis seperti pembangunan kasino, galangan kapal hingga kilang minyak.
"Saya tahu sendiri, banyak LSM didanai dari luar negeri buat demo di Indonesia. Kalau negara ini buka kilang minyak atau mau buka kasino, ada demo. Katanya isu lingkungan, padahal bisnis mereka yang terganggu. Singapura misalnya, mereka pintar. Setiap orang main kasino, negara dapat 25 persen royalti," katanya.
Dirinya menyatakan, pendekatan realistis dan berbasis bisnis perlu diterapkan dalam membuat kebijakan. Bukan hanya soal moral atau agama tapi manfaat ekonomi yang lebih luas.
"Kalau daerah seperti Pangkal Pinang APBD-nya cuma Rp1 triliun dengan pendapatan daerah cuma Rp100 miliar, itu artinya 90 persen hidup dari belas kasihan pusat. Kalau ini perusahaan, sudah bangkrut. Kenapa kita tidak bikin industri yang masuk akal? Yang bisa jalan sekarang ya pariwisata, hiburan, termasuk kasino fisik dengan regulasi yang ketat," katanya.
Dia mengatakan, bangsa Indonesia harus berhenti menjadi mesin uang bagi negara lain dan mulai mengelola sendiri potensi ekonominya.
Faktanya orang-orang kaya atau konglomerat Indonesia butuh hiburan. Faktanya mereka buang uang di luar negeri. Kalau negara bisa ambil alih ini secara legal, ini bukan hanya pemasukan negara, tapi bentuk kedaulatan, katanya.""", 0),





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




    ("Kemendikdasmen Pastikan Tak Ada Kasus Jual Beli Kursi SPMB di Bandung", """Wakil Menteri Pendidikan Dasar dan Menengah Atip Latipulhayat membantah adanya kasus dugaan jual beli kursi di Sistem Penerimaan Murid Baru (SPMB) 2025 di Bandung, Jawa Barat. "Enggak (ada), justru kami sudah melakukan investigasi dan di Bandung itu bukan kecurangan," kata Atip usai menerima Gubernur Jawa Barat Dedi Mulyadi di kantor Kemendikdasmen, Jakarta, Selasa (17/6/2025). Atip menyampaikan, Kemendikdasmen telah mengonfirmasi terkait hal tersebut kepada Wali Kota Bandung, Muhammad Farhan. "Jadi di Bandung sendiri itu tidak ada (kasus jual beli kursi SPMB)," kata Atip menegaskan.Pemkot Bandung menerima peringatan dari Saber Pungli Kota Bandung terkait dugaan jual beli kursi di empat SMP Negeri, dengan penawaran mencapai Rp 5 juta hingga Rp 8 juta per kursi. "Ada empat (sekolah) SMP ya," kata Dani di Balai Kota Bandung, Jalan Wastukencana, Kota Bandung, Selasa (10/6/2025). Dani menambahkan, apabila dugaan praktik jual beli kursi sekolah benar-benar terjadi, yang akan dipidana adalah pihak yang menawarkan dan pihak yang bersedia membayar.""", 0),




    ("Kejagung Sita Rp 11,8 Triliun Uang yang Dikembalikan Wilmar Group", """Kejaksaan Agung menyita Rp 11.880.351.802.619, yang merupakan penyerahan dari lima terdakwa korporasi dalam Wilmar Group terkait kasus korupsi ekspor crude palm oil (CPO). “Bahwa dalam perkembangan lima terdakwa korporasi tersebut mengembalikan uang kerugian negara yang ditimbulkannya, yaitu Rp 11.880.351.802.619,” ujar Direktur Penuntutan Kejaksaan Agung, Sutikno dalam konferensi pers di Gedung Bundar Jampidsus, Kejaksaan Agung, Jakarta, Selasa (17/6/2025). Sutikno mengatakan, uang yang dikembalikan oleh Wilmar Group ini langsung disita oleh penyidik dan dimasukkan dalam rekening penampungan Jampidsus. Uang yang dikembalikan ini merupakan hasil kerugian negara yang dihitung oleh Badan Pengawasan Keuangan dan Pembangunan (BPKP). Barang bukti yang telah disita juga dimaksudkan ke memori kasasi karena perkara ini tengah berproses di Mahkamah Agung. Uang yang dikembalikan ini merupakan hasil kerugian negara yang dihitung oleh Badan Pengawasan Keuangan dan Pembangunan (BPKP). Barang bukti yang telah disita juga dimaksudkan ke memori kasasi karena perkara ini tengah berproses di Mahkamah Agung. Sementara, dikutip dari keterangan resmi Kejaksaan Agung, JPU menuntut para terdakwa untuk membayarkan sejumlah denda dan denda pengganti. Terdakwa PT Wilmar Group dituntut untuk membayar denda sebesar Rp 1 miliar dan uang pengganti sebesar Rp 11.880.351.802.619. Jika uang ini tidak dibayarkan, harta Tenang Parulian selaku Direktur dapat disita dan dilelang, apabila tidak mencukupi terhadap Tenang Parulian dikenakan subsidiair pidana penjara 19 tahun. Terdakwa Permata Hijau Group dituntut untuk membayar denda sebesar Rp 1 miliar dan uang pengganti sebesar Rp 937.558.181.691,26. Jika uang ini tidak dibayarkan, harta David Virgo selaku pengendali lima korporasi di dalam Permata Hijau Group dapat disita untuk dilelang, apabila tidak mencukupi terhadap David Virgo dikenakan subsidiair penjara selama 12 bulan.""", 0),




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

def test_only_title_batch():
    correct = 0
    total = len(test_data)
    print("\n[TITLE ONLY] Prediction Results:")
    for i, (title, _, label) in enumerate(test_data):
        hasil, _ = classify(title, '', model_option="XGBoost")
        pred = 1 if hasil == "HOAX" else 0
        status = "✅" if pred == label else "❌"
        print(f"{status} {i+1}. Label: {label}, Predicted: {pred} ({hasil})")
        if pred == label:
            correct += 1
    print(f"[TITLE ONLY] Correct predictions: {correct}/{total}")
    assert correct >= 5

def test_only_body_batch():
    correct = 0
    total = len(test_data)
    print("\n[BODY ONLY] Prediction Results:")
    for i, (_, body, label) in enumerate(test_data):
        hasil, _ = classify('', body, model_option="XGBoost")
        pred = 1 if hasil == "HOAX" else 0
        status = "✅" if pred == label else "❌"
        print(f"{status} {i+1}. Label: {label}, Predicted: {pred} ({hasil})")
        if pred == label:
            correct += 1
    print(f"[BODY ONLY] Correct predictions: {correct}/{total}")
    assert correct >= 5

def test_combined_input_batch():
    correct = 0
    total = len(test_data)
    print("\n[COMBINED INPUT] Prediction Results:")
    for i, (title, body, label) in enumerate(test_data):
        hasil, _ = classify(title, body, model_option="XGBoost")
        pred = 1 if hasil == "HOAX" else 0
        status = "✅" if pred == label else "❌"
        print(f"{status} {i+1}. Label: {label}, Predicted: {pred} ({hasil})")
        if pred == label:
            correct += 1
    print(f"[COMBINED INPUT] Correct predictions: {correct}/{total}")
    assert correct >= 7
