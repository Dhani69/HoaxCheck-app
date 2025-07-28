import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi factory Sastrawi sekali saja
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def clean_text(text: str) -> str:
    """
    Membersihkan teks dari URL, karakter tidak penting,
    lowercase, hapus angka, hapus stopword, dan stemming.
    """
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Hapus karakter selain huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Case folding
    text = text.lower()
    # Hapus angka 
    text = re.sub(r'\d+', '', text)
    # Hapus whitespace berlebih 
    text = re.sub(r'\s+', ' ', text).strip()
    # Stopword removal
    text = stopword_remover.remove(text)
    # Stemming
    text = stemmer.stem(text)

    return text
