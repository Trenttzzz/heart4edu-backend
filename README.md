# CPR Inference API (FastAPI + ONNX Runtime)

Layanan FastAPI untuk inferensi kualitas kompresi CPR berbasis model ONNX. Mendukung:
- HTTP endpoint untuk single/batch prediksi
- Ingest streaming nilai kedalaman (buffer 10 nilai) lalu broadcast hasil via WebSocket
- UI mini untuk memantau hasil inferensi per session_id

## Prasyarat
- Python 3.12+
- File model dan scaler:
  - `model/model_cpr.onnx`
  - `model/scaler_params.json` (berisi mean & scale 10 dimensi)

Contoh struktur folder:
```
backend/
├─ app.py
├─ model/
│  ├─ model_cpr.onnx
│  └─ scaler_params.json
```

Format `scaler_params.json` (contoh):
```json
{
  "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
```
Catatan: Panjang array harus 10.

## Instalasi & Menjalankan
1) Buat virtual environment dan aktifkan
- Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
- macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependency
```bash
pip install fastapi "uvicorn[standard]" onnxruntime numpy anyio pydantic
```

3) Pastikan file `model/model_cpr.onnx` dan `model/scaler_params.json` tersedia.

4) Jalankan server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Saat startup, console akan menampilkan IP lokal, contoh:
```
===> FastAPI reachable at: http://192.168.1.10:8000
```

5) Akses dari perangkat lain dalam satu Wi‑Fi/LAN dengan IP tersebut, misal:
- Health check: http://192.168.1.10:8000/health
- UI mini (WebSocket): http://192.168.1.10:8000/ui?session_id=default

## Endpoint Utama
- GET `/health` — cek status.
- POST `/predict` — prediksi untuk 1 window (10 nilai).
- POST `/predict/batch` — prediksi untuk banyak window (B x 10).
- POST `/ingest?mode=sliding` — kirim 1 nilai; setelah 10 nilai per session, otomatis inferensi, dan akan terus inferensi karena menggunakan konsep sliding window.
- POST `/ingest?mode=nonoverlap` — kirim 1 nilai; setelah 10 nilai per session, otomatis inferensi, kemudian clear buffer dan menunggu hingga nilai total buffer 10.
- GET `/last/{session_id}` — ambil hasil terakhir per session.
- WS `/ws/{session_id}` — stream hasil inferensi.
- UI `/ui?session_id=...` — halaman monitor sederhana.

## Contoh Pengujian Cepat

### 1) Health
```bash
curl http://localhost:8000/health
```

### 2) Prediksi Single Window (10 nilai)
```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{ \"session_id\":\"default\", \"depth_cm\":[5,5,5,5,5,5,5,5,5,5] }"
```
Respons contoh:
```json
{
  "class_index": 0,
  "class_label": "stabil",
  "probs": [0.6, 0.1, 0.15, 0.15]
}
```

### 3) Prediksi Batch (B window)
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"default",
    "batches":[
      [5,5,5,5,5,5,5,5,5,5],
      [4,4,4,4,4,4,4,4,4,4]
    ]
  }'
```

### 4) Ingest Streaming (memicu inferensi setelah 10 nilai)
- Kirim 10 nilai ke session `default`:
- Bash:
```bash
for i in {1..10}; do
  curl -s -X POST "http://localhost:8000/ingest" \
    -H "Content-Type: application/json" \
    -d '{"session_id":"default","depth_cm":5.0}' | jq .
done
```
- PowerShell:
```powershell
1..10 | ForEach-Object {
  curl -Method POST "http://localhost:8000/ingest" `
    -ContentType "application/json" `
    -Body '{"session_id":"default","depth_cm":5.0}'
}
```
Pada kiriman ke-10, field `"inferred": true` dan `"result"` terisi. Hasil juga dibroadcast ke WebSocket.

### 5) Ambil Hasil Terakhir
```bash
curl http://localhost:8000/last/default
```

### 6) WebSocket Monitor (UI)
Buka di browser:
```
http://localhost:8000/ui?session_id=default
```
Kirim data via `/ingest` (seperti di atas). UI akan menampilkan label, index, dan probabilitas setelah 10 nilai diterima.

## Pengujian via Python (opsional)
```python
import requests

base = "http://localhost:8000"
print(requests.get(f"{base}/health").json())

payload = {
  "session_id": "default",
  "depth_cm": [5,5,5,5,5,5,5,5,5,5]
}
print(requests.post(f"{base}/predict", json=payload).json())
```

## Catatan & Troubleshooting
- File model/scaler wajib ada:
  - Jika `model/scaler_params.json` hilang atau panjang array bukan 10, server akan error saat start atau inferensi.
- Provider ONNX: menggunakan CPU (`CPUExecutionProvider`), tidak butuh CUDA.
- CORS sudah diizinkan untuk semua origin, cocok untuk uji front-end lokal.
- Akses dari perangkat lain gunakan IP yang dicetak saat startup, contoh `http://192.168.x.x:8000`.
- Window input **(jika mode = sliding)** selalu 10 nilai. `/predict` butuh array panjang 10. `/ingest?mode=sliding` akan infer setelah 10 nilai diterima per `session_id`.
- Jika menggunakan `/ingest?mode=nonoverlap` maka /predict akan dipanggil setiap kali 10 input nilai baru, tidak selalu run inference setelah 10 nilai.
