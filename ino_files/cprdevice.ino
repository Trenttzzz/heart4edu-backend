#include <WiFi.h>
#include <HTTPClient.h>

int trigPin = 5;
int echoPin = 18;

// isi ssid wifi dan password dibawah
const char* ssid = "fake_username";
const char* password = "fake_password";

// ada 2 mode -> ?mode=sliding & ?mode=nonoverlap 
// default mode = sliding
const char* SERVER_INGEST = "http://192.168.1.5:8000/ingest?mode=nonoverlap"; 
const char* SESSION_ID    = "esp32-sesi-01";

// ====== Peak detection params (versi tahan double-peak) ======
const bool  PEAK_IS_MIN        = true;     // makin dekat = makin dalam
const float VEL_ON_CM          = 0.25f;    // mulai TURUN (|vel| cm per sampel)
const float VEL_OFF_CM         = 0.15f;    // konfirmasi BALIK (harus < VEL_ON)
const uint8_t REVERSE_CONFIRM  = 2;        // butuh N sampel beruntun arah balik
const float AMP_THRESH_CM      = 1.0f;     // amplitudo minimal dihitung peak
const float REARM_DEPTH_CM     = 0.8f;     // harus dekat baseline utk start baru
const uint16_t SAMPLE_MS       = 8;        // ~125 Hz
const uint16_t REFRACT_MS      = 350;      // jeda anti-peak ganda
const uint16_t MIN_STROKE_MS   = 80;       // minimal durasi turun
const float  EMA_ALPHA         = 0.25f;    // smoothing

// ====== State ======
enum Phase { IDLE, DOWN, UP };
Phase phase = IDLE;

float emaVal = NAN, lastVal = NAN;
float startStrokeVal = NAN, extremeVal = NAN;
float baselineRest = NAN;

uint32_t lastSampleAt = 0, lastPeakAt = 0, strokeStartAt = 0;
uint8_t  reverseCount = 0;

// ====== Helpers ======
float readUltrasonicCM() {
  digitalWrite(trigPin, LOW);  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH); delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  // timeout 30 ms
  unsigned long us = pulseIn(echoPin, HIGH, 30000UL);
  if (us == 0) return NAN;
  float d = us * 0.034f / 2.0f;
  if (d <= 0 || d > 500.0f) return NAN;
  return d;
}

float ema(float x) {
  if (isnan(emaVal)) emaVal = x;
  else emaVal = EMA_ALPHA * x + (1.0f - EMA_ALPHA) * emaVal;
  return emaVal;
}

float depthFromRaw(float raw) {
  if (isnan(baselineRest)) return NAN;
  // PEAK_IS_MIN=true -> depth = baseline - jarak
  return max(0.0f, baselineRest - raw);
}

void sendPeak(float depth_cm) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http; http.setTimeout(5000);
  if (!http.begin(SERVER_INGEST)) return;
  http.addHeader("Content-Type", "application/json");
  String json = "{\"session_id\":\"" + String(SESSION_ID) + "\",\"depth_cm\":" + String(depth_cm,2) + "}";
  int code = http.POST(json);
  if (code > 0) {
    Serial.printf("POST /ingest -> %d\n", code);
    Serial.println(http.getString());
  } else {
    Serial.printf("HTTP POST failed: %s (%d)\n", http.errorToString(code).c_str(), code);
  }
  http.end();
}

// ====== Setup ======
void setup() {
  Serial.begin(115200);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(200); Serial.print("."); }
  Serial.println("\nWiFi connected.");

  // Kalibrasi baseline (jangan ditekan)
  Serial.println("Kalibrasi baseline, jangan tekan...");
  const int CALIB_SAMPLES = 150;
  int got = 0; double sum = 0;
  while (got < CALIB_SAMPLES) {
    float r = readUltrasonicCM();
    if (!isnan(r)) { sum += r; got++; }
    delay(SAMPLE_MS);
  }
  baselineRest = sum / got;
  emaVal = baselineRest;
  lastVal = emaVal;
  Serial.print("Baseline (cm): "); Serial.println(baselineRest, 2);
}

// ====== Loop ======
void loop() {
  if (millis() - lastSampleAt < SAMPLE_MS) return;
  lastSampleAt = millis();

  float raw = readUltrasonicCM();
  if (isnan(raw)) return;

  float y = ema(raw);
  float vel;
  if (isnan(lastVal)) vel = 0;
  else vel = y - lastVal;
  lastVal = y;

  float depth = depthFromRaw(y);
  bool inRefract = (millis() - lastPeakAt < REFRACT_MS);

  switch (phase) {

    case IDLE: {
      // re-arm hanya jika dekat baseline dan tidak dalam refractory
      if (!inRefract && depth <= REARM_DEPTH_CM) {
        bool startDown = PEAK_IS_MIN ? (vel <= -VEL_ON_CM) : (vel >= VEL_ON_CM);
        if (startDown) {
          phase = DOWN;
          startStrokeVal = y;
          extremeVal = y;
          strokeStartAt = millis();
          reverseCount = 0;
        }
      }
      break;
    }

    case DOWN: {
      // track titik paling dalam
      if (PEAK_IS_MIN) { if (y < extremeVal) extremeVal = y; }
      else             { if (y > extremeVal) extremeVal = y; }

      // deteksi balik arah dengan hysteresis + konfirmasi N sampel
      bool upDir = PEAK_IS_MIN ? (vel >= VEL_OFF_CM) : (vel <= -VEL_OFF_CM);
      if (upDir) reverseCount++; else reverseCount = 0;

      bool enoughTime = (millis() - strokeStartAt >= MIN_STROKE_MS);
      if (reverseCount >= REVERSE_CONFIRM && enoughTime) {
        float amp = fabsf(extremeVal - startStrokeVal);
        float depth_cm = depthFromRaw(extremeVal);

        lastPeakAt = millis();  // aktifkan refractory (walau peak ditolak)

        if (!isnan(depth_cm) && amp >= AMP_THRESH_CM) {
          Serial.print("PEAK depth (cm): "); Serial.println(depth_cm, 2);
          sendPeak(depth_cm);
        } else {
          Serial.println("Peak diabaikan (amplitudo kecil/invalid).");
        }

        phase = UP;              // lanjut fase naik
        reverseCount = 0;
      }
      break;
    }

    case UP: {
      // tunggu hingga kembali dekat baseline & refractory selesai → siap siklus baru
      if (!inRefract && depth <= REARM_DEPTH_CM) {
        phase = IDLE;
      }
      break;
    }
  }
}
