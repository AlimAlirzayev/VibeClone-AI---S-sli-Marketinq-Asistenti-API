import re
import tiktoken
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./vibeclone.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 1. İSTİFADƏÇİ PROFİLİ (n8n MCP-dən gələn real LinkedIn datası bura yazılacaq)
class UserProfile(Base):
    __tablename__ = "user_profiles"

    telegram_id = Column(String(100), primary_key=True, index=True)
    linkedin_url = Column(String(200), nullable=False)
    writing_style_context = Column(Text, nullable=True) # MCP-nin çəkdiyi real postlar bura düşəcək

    # Əlaqə: Bir istifadəçinin çoxlu postu ola bilər
    posts = relationship("VibePost", back_populates="owner")

# 2. YARADILAN POSTLAR CƏDVƏLİ
class VibePost(Base):
    __tablename__ = "vibe_posts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), ForeignKey("user_profiles.telegram_id"), nullable=False)
    raw_transcript = Column(Text, nullable=False)
    clean_text = Column(Text, nullable=True)
    token_count = Column(Integer, nullable=True)
    generated_post = Column(Text, nullable=True)

    owner = relationship("UserProfile", back_populates="posts")

# --- DATA EMALI ---
def clean_transcript(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\b(ııı|şey|yəni|eee|baxın)\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def init_db() -> None:
    Base.metadata.create_all(bind=engine)