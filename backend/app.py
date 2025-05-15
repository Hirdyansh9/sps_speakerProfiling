from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pydub import AudioSegment
from utils import load_model, predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = 'model/model.pth'

@app.post("/predict/")
async def get_prediction(audio: UploadFile = File(...)):
    # Save the uploaded WebM file temporarily
    temp_input_path = f"temp_{audio.filename}"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Convert WebM to WAV
    temp_wav_path = "temp_recording.wav"
    try:
        audio_segment = AudioSegment.from_file(temp_input_path)  # Load WebM
        audio_segment.export(temp_wav_path, format="wav")  # Export as WAV

        # Run prediction on the WAV file
        prediction = predict(model, temp_wav_path)
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    return prediction