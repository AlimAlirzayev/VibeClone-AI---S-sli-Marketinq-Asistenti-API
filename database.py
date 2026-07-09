print("FAYL TAMAMİLƏ UĞURLA İCRA OLUNUR!")

import re
import tiktoken
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Windows mühitində bloklanmamaq üçün SQLite istifadə edirik.
# MacBook-da PostgreSQL-ə keçəndə sadəcə bu linki yeniləyəcəksən.
DATABASE_URL = "sqlite:///./vibeclone.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- SQLALCHEMY MODELİ (Məlumat Strukturu) ---
class VibePost(Base):
    __tablename__ = "vibe_posts"

    id: int = Column(Integer, primary_key=True, index=True)
    raw_transcript: str = Column(Text, nullable=False)   # Telegram-dan gələn xam səs mətni
    clean_text: str = Column(Text, nullable=True)        # Regex ilə təmizlənmiş mətn
    token_count: int = Column(Integer, nullable=True)     # tiktoken tərəfindən sayılan tokenlər
    generated_post: str = Column(Text, nullable=True)    # Gemini-nin çıxardığı son LinkedIn postu

# --- DATA EMALI NÜVƏSİ (Cari Ayın Mövzuları) ---

def clean_transcript(text: str) -> str:
    """ Mətndəki HTML teqlərini və parazit danışıq sözlərini Regex ilə təmizləyir. """
    text = re.sub(r"<[^>]+>", "", text)  # HTML teqlərini təmizlə
    # Azərbaycan dilindəki tipik parazit sözləri təmizləyirik
    text = re.sub(r"\b(ııı|şey|yəni|eee|baxın)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def count_tokens(text: str) -> int:
    """ tiktoken istifadə edərək mətndəki dəqiq token sayını hesablayır. """
    encoding = tiktoken.get_encoding("cl100k_base")  # Müasir LLM-lərin standart tokenizer mühərriki
    return len(encoding.encode(text))

# Bazanı və cədvəlləri lokal olaraq yaratmaq üçün funksiya
def init_db() -> None:
    Base.metadata.create_all(bind=engine)


    if __name__ == "__main__":
        print("Infrastruktur qurulur: SQLite bazası və cədvəllər yaradılır...")
        init_db()
        print("Uğurlu! 'vibeclone.db' faylı layihə qovluğunda yaradıldı.")

        # Kiçik bir Regex və tiktoken testi (Sanity Check)
        test_text = "ııı Salam, bu bir <p>marketinq</p> şey postudur yəni."
        clean = clean_transcript(test_text)
        tokens = count_tokens(clean)
        print(f"Test Mətn: {clean} | Token Sayı: {tokens}")