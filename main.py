from fastapi import FastAPI, File, UploadFile
from transformers import pipeline

app = FastAPI()
model = pipeline("audio-classification", model="speechbrain/antispoofing-celeb-v2")

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    result = model(audio.file)
    return {"label": result[0]['label'], "score": float(result[0]['score'])}
