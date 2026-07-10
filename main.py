import os
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import database

load_dotenv()

app = FastAPI(title="VibeClone AI Engine", version="2026.2.0")


@app.on_event("startup")
def on_startup():
    database.init_db()


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- LANGCHAIN LCEL ZƏNCİRİ ---
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Prompt artıq dinamikdir: həm səsi, həm də istifadəçinin stilini qəbul edir
prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "Sən professional LinkedIn asistentisən. Sənin vəzifən istifadəçinin səs qeydini "
        "onun real yazı üslubuna uyğunlaşdıraraq cəlbedici bir LinkedIn postuna çevirməkdir.\n\n"
        "İstifadəçinin Şəxsi Yazı Üslubu (Keçmiş Postları):\n"
        "{style_context}\n\n"
        "Qaydalar: Yazı tam təbii olmalı və yuxarıdakı üsluba bənzəməlidir."
    )),
    ("user", "Səs qeydimin mətni: {transcript}")
])

copywriter_chain = prompt_template | model | StrOutputParser()


# --- PYDANTIC MODELLƏRİ ---
class RegisterUser(BaseModel):
    telegram_id: str
    linkedin_url: str
    scraped_context: str  # n8n MCP-dən gələn xülasə


class VoiceInput(BaseModel):
    telegram_id: str
    raw_transcript: str


# --- ENDPOINTLƏR ---

@app.post("/api/v1/register-style", status_code=status.HTTP_201_CREATED)
def register_user_style(payload: RegisterUser, db: Session = Depends(get_db)):
    """ n8n MCP LinkedIn-dən datanı çəkib bu endpointə vuracaq """
    user = db.query(database.UserProfile).filter(database.UserProfile.telegram_id == payload.telegram_id).first()

    if user:
        user.writing_style_context = payload.scraped_context
    else:
        user = database.UserProfile(
            telegram_id=payload.telegram_id,
            linkedin_url=payload.linkedin_url,
            writing_style_context=payload.scraped_context
        )
        db.add(user)

    db.commit()
    return {"message": "İstifadəçi stili uğurla yadda saxlanıldı!"}


@app.post("/api/v1/generate-post")
def process_voice_input(payload: VoiceInput, db: Session = Depends(get_db)):
    """ Telegram-dan səs gələndə işə düşən əsas mühərrik """
    # 1. İstifadəçini bazadan tapırıq
    user = db.query(database.UserProfile).filter(database.UserProfile.telegram_id == payload.telegram_id).first()
    if not user or not user.writing_style_context:
        raise HTTPException(status_code=404, detail="Əvvəlcə LinkedIn hesabınızı bağlayın!")

    # 2. Mətni təmizləyirik
    cleaned_text = database.clean_transcript(payload.raw_transcript)
    token_count = database.count_tokens(cleaned_text)

    # 3. LangChain ilə Generasiya (Stil + Yeni Səs)
    try:
        ai_generated_post = copywriter_chain.invoke({
            "style_context": user.writing_style_context,
            "transcript": cleaned_text
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI Xətası: {str(e)}")

    # 4. Tarixçəni yazırıq
    db_entry = database.VibePost(
        user_id=user.telegram_id,
        raw_transcript=payload.raw_transcript,
        clean_text=cleaned_text,
        token_count=token_count,
        generated_post=ai_generated_post
    )
    db.add(db_entry)
    db.commit()

    return {"generated_post": ai_generated_post, "tokens_used": token_count}