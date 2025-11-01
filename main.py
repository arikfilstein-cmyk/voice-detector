from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import uvicorn
import iם
import soundfile as sf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    global model
    print("⏳ Loading model... please wait.")
    model = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
    print("✅ Model loaded successfully!")

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    try:
        # קריאת הקובץ
        audio_bytes = await audio.read()
        data, samplerate = sf.read(io.BytesIO(audio_bytes))

        # הפעלת המודל
        result = model(io.BytesIO(audio_bytes))

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
