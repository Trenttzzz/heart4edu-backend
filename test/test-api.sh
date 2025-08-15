# kirim 9 kali dulu
for i in {1..9}; do curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"session_id":"esp32-sesi-01","depth_cm":3.2}'; echo; done

# kirim ke-10 -> respons harus "inferred: true" + ada "result"
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"session_id":"esp32-sesi-01","depth_cm":3.5}'
