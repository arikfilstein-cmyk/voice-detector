from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn

app = FastAPI()

# מאפשר גישה מכל מקור (כדי שהממשק שלך ב-Lovable יוכל לקרוא)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# טוען את המודל לזיהוי קול מזויף
model = pipeline("audio-classification", model="speechbrain/antispoofing-celeb-v2")

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    result = model(audio.file)
    label = result[0]['label']
    score = round(float(result[0]['score']), 3)
    return {"label": label, "score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
