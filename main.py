from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import tempfile
from sentence_transformers import SentenceTransformer, util
from normalize import load_kamus_slang, normalize
import csv
import torch

app = FastAPI()

# Load model
stt_model = whisper.load_model("base")  # whisper base model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load kamus slang
kamus = load_kamus_slang("kamus_slang.csv")

# Load dataset
def load_dataset_dengan_label(path):
    kalimat_krisis, kalimat_non_krisis = [], []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            teks = row['text'].strip().lower()
            label = row['label'].strip()
            if label == '1':
                kalimat_krisis.append(teks)
            else:
                kalimat_non_krisis.append(teks)
    return kalimat_krisis, kalimat_non_krisis

kalimat_krisis, kalimat_non_krisis = load_dataset_dengan_label("dataset_gabungan.csv")
emb_krisis = sbert_model.encode(kalimat_krisis, convert_to_tensor=True)
emb_non_krisis = sbert_model.encode(kalimat_non_krisis, convert_to_tensor=True)

# Fungsi bagi teks jika lebih dari 250 kata
def bagi_teks(teks, batas_kata=250):
    kata = teks.strip().split()
    return [' '.join(kata[i:i+batas_kata]) for i in range(0, len(kata), batas_kata)]

# API Endpoint
@app.post("/cek-krisis/")
async def cek_krisis(file: UploadFile = File(...)):
    try:
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Speech to text
        result = stt_model.transcribe(tmp_path)
        input_user = result['text'].strip()

        # Potong teks jika panjang
        input_parts = bagi_teks(input_user) if len(input_user.split()) > 250 else [input_user]
        input_parts_normalized = [normalize(part.lower(), kamus) for part in input_parts]

        # Cek krisis
        threshold = 0.75
        for part in input_parts_normalized:
            emb_input = sbert_model.encode(part, convert_to_tensor=True)
            sim_krisis = util.cos_sim(emb_input, emb_krisis)[0]
            sim_non_krisis = util.cos_sim(emb_input, emb_non_krisis)[0]

            skor_krisis = torch.max(sim_krisis).item()
            skor_non_krisis = torch.max(sim_non_krisis).item()

            if skor_krisis > skor_non_krisis and skor_krisis > threshold:
                return JSONResponse(content={"is_krisis": True})

        return JSONResponse(content={"is_krisis": False})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
