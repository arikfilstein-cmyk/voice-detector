from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import numpy as np
import io

app = FastAPI(title="Voice Detector API", version="1.1")

# === עוזר פנימי לחישוב מאפייני קול ===
def extract_audio_features(file_bytes):
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    energy = np.mean(samples ** 2)
    avg_amplitude = np.mean(np.abs(samples))
    return {
        "duration_sec": len(audio) / 1000,
        "energy": float(energy),
        "amplitude": float(avg_amplitude)
    }

# === ברירת מחדל ===
@app.get("/")
async def root():
    return {"message": "Voice Detector API is running. Visit /docs to use the interface."}

# === ניתוח קול יחיד (מה שכבר עבד) ===
@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    features = extract_audio_features(audio_bytes)
    fake_score = np.clip(features["energy"] * 1000 % 100, 0, 100)
    verdict = "Real" if fake_score < 50 else "Fake"
    return {"verdict": verdict, "score": round(fake_score, 2), "features": features}

# === השוואת שני קבצי קול ===
@app.post("/compare")
async def compare_voices(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    # קריאת שני הקבצים
    audio1 = await file1.read()
    audio2 = await file2.read()

    # הפקת מאפיינים
    f1 = extract_audio_features(audio1)
    f2 = extract_audio_features(audio2)

    # חישוב דמיון פשוט
    amp_diff = abs(f1["amplitude"] - f2["amplitude"])
    energy_diff = abs(f1["energy"] - f2["energy"])
    similarity = max(0, 100 - (amp_diff * 50000 + energy_diff * 1e6))

    return {
        "similarity_percent": round(similarity, 2),
        "file1_features": f1,
        "file2_features": f2
    }

# === פרטי קובץ ===
@app.post("/metadata")
async def audio_metadata(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    features = extract_audio_features(audio_bytes)
    return {"metadata": features}
