from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import os
import tempfile

app = FastAPI()

# קוראים את הטוקן מהסביבה (הכנסת אותו ב-Render תחת HF_TOKEN)
HF_TOKEN = os.getenv("HF_TOKEN")

# טוענים את המודל עם אימות (חשוב!)
model = pipeline(
    task="audio-classification",
model="audeering/antispoofing-celeb-v2"
    use_auth_token=HF_TOKEN,  # זה קו המפתח
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # שומרים זמנית את הקובץ לדיסק
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # מריצים חיזוי
    preds = model(tmp_path)

    # מחזירים תוצאה פשוטה: התווית עם הציון הגבוה ביותר
    best = max(preds, key=lambda p: p.get("score", 0))
    return {"label": best.get("label"), "score": float(best.get("score", 0.0))}
