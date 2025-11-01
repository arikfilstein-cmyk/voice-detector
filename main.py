from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import io
import soundfile as sf
import uvicorn

app = FastAPI(
    title="Voice Detector API",
    description="Detects whether an audio file is Bonafide (real) or Spoofed (fake).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    global model
    print("🔄 Loading model, please wait...")
    model = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
    print("✅ Model loaded successfully!")

@app.get("/")
def root():
    return {"message": "Voice Detector API is running. Visit /docs to use the interface."}

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    result = model(io.BytesIO(audio_bytes))
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
