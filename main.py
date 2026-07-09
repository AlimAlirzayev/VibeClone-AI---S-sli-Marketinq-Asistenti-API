import os
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from dotenv import load_dotenv

# LangChain building blocks (Cari ayın mövzusu)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import database

# .env faylındakı dəyişənləri oxuyuruq
load_dotenv()

app = FastAPI(
    title="VibeClone AI - Səsli Marketinq Asistenti API",
    description="Səs yazılarından şəxsi üslubda LinkedIn kontenti yaradan agent tətbiqi.",
    version="2026.1.0"
)

@app.on_event("startup")
def on_startup():
    database.init_db()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- LANGCHAIN LCEL ZƏNCİRİNİN QURULMASI ---

# 1. Müasir Gemini modelinin çağırılması (2026 standartı)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 2. Advanced Prompt Engineering (LinkedIn-də vizual fərq yaradacaq marketinq strukturu)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "Sən professional bir LinkedIn Growth Hacker və Creative Copywriter-sən. "
        "Sənə istifadəçinin təbii və xaotik səs yazısının təmizlənmiş mətni veriləcək. "
        "Sənin vəzifən bu mətndəki əsas ideyanı götürüb, oxuyanda LinkedIn-də 'Wow' effekti "
        "yaradacaq, yüksək reaksiya (engagement) toplayacaq bir post hazırlamaqdır.\n\n"
        "Qaydalar:\n"
        "- Yazı dili tamamilə təbii, axıcı və peşəkar Azərbaycan dilində olmalıdır.\n"
        "- Struktur mütləq cəlbedici bir başlıq (Hook) ilə başlamalı, abzaslar arası boşluqlar olmalıdır.\n"
        "- Darıxdırıcı korporativ dildən uzaq dur, insani duyğunu qoru.\n"
        "- Sonda 2-3 ədəd trend sahə heşteqi əlavə et."
    )),
    ("user", "Mənim səs yazımın mətni budur:\n\n{transcript}")
])

# 3. LCEL Chain Konstruksiyası (Prompt -> Model -> Parser)
# Kursun ən vacib cari ay mövzusu!
copt_writer_chain = prompt_template | model | StrOutputParser()


# --- PYDANTIC MODELLƏRİ ---
class TranscriptInput(BaseModel):
    raw_transcript: str

class PostResponse(BaseModel):
    id: int
    clean_text: str
    token_count: int
    generated_post: str

    class Config:
        from_attributes = True

# --- API ENDPOINT ---

@app.post("/api/v1/process-voice", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
def process_voice_input(payload: TranscriptInput, db: Session = Depends(get_db)):
    if not payload.raw_transcript.strip():
        raise HTTPException(status_code=400, detail="Səs yazısının mətni boş ola bilməz!")

    # 1. Data Hazırlığı: Regex ilə təmizləmə (Dərs 13)
    cleaned_text = database.clean_transcript(payload.raw_transcript)

    # 2. tiktoken ilə token sayımı (Dərs 13)
    token_count = database.count_tokens(cleaned_text)

    # 3. Real LangChain LCEL Zəncirinin işə salınması
    try:
        ai_generated_post = copt_writer_chain.invoke({"transcript": cleaned_text})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API ilə rabitə xətası: {str(e)}")

    # 4. Məlumatın verilənlər bazasına yazılması (Dərs 12)
    db_entry = database.VibePost(
        raw_transcript=payload.raw_transcript,
        clean_text=cleaned_text,
        token_count=token_count,
        generated_post=ai_generated_post
    )

    try:
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        return db_entry
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Baza transaction xətası: {str(e)}")

@app.get("/api/v1/history", response_model=List[PostResponse])
def get_history(db: Session = Depends(get_db)):
    return db.query(database.VibePost).all()