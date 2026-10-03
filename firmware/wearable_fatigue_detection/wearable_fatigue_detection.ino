/*
   ============================================================
   WEARABLE FATIGUE DETECTION SYSTEM
   ============================================================

   ESP32 + MAX30102 + MPU6050 + AD8232 ECG
   + LED + Buzzer + Button + WiFi + ThingSpeak
   + Optional Firebase + Telegram

   ------------------------------------------------------------
   AD8232 ECG WIRING
   ------------------------------------------------------------

   AD8232 OUTPUT  -> ESP32 GPIO 34
   AD8232 LO+     -> ESP32 GPIO 35
   AD8232 LO-     -> ESP32 GPIO 32
   AD8232 3.3V    -> ESP32 3.3V
   AD8232 GND     -> ESP32 GND

   ------------------------------------------------------------
   OTHER PINS
   ------------------------------------------------------------

   MAX30102 SDA -> GPIO 21
   MAX30102 SCL -> GPIO 22

   MPU6050 SDA -> GPIO 21
   MPU6050 SCL -> GPIO 22

   LED          -> GPIO 4
   BUZZER       -> GPIO 5
   BUTTON       -> GPIO 33  (MOMENTARY only — a push button or a brief
                             wire touch. Do NOT wire this permanently to
                             GND, or ecgMarked will read true 100% of the
                             time instead of only when you press it.)

   NEW STATUS LEDS (added on top of the existing wiring, no other pins changed):
   FINGER LED   -> GPIO 25 -> 220Ω resistor -> LED -> GND
                   lights when MAX30102 detects a finger
   ECG LED      -> GPIO 26 -> 220Ω resistor -> LED -> GND
                   lights when ECG leads-off clears (good skin contact)

   ------------------------------------------------------------
   SEEING THE ECG WAVEFORM AND THE TEXT DEBUG OUTPUT TOGETHER
   ------------------------------------------------------------

   Arduino IDE 2.x lets Serial Monitor and Serial Plotter both stay open
   at the same time, reading the same stream. This sketch now always
   prints BOTH the numeric ECG plot line (every ~4ms) AND the readable
   debug block (once per second) — no mode switch needed.

     1. Upload this sketch.
     2. Open Tools -> Serial Plotter, set baud to 115200. With electrodes
        on and leads-off cleared, you'll see three traces: raw ECG
        signal, the adaptive baseline, and a peak marker that spikes
        briefly on each detected heartbeat. If the marker doesn't land
        on visible R-wave peaks, tune ECG_PEAK_THRESHOLD below.
     3. Also open Tools -> Serial Monitor (a separate window/panel) at
        115200. You'll see the same stream as text — mostly a fast
        scroll of numeric plot lines, with a readable debug block
        (BPM, WiFi, temp, etc.) appearing once per second among them.
        The plotter ignores those text lines automatically.

   ------------------------------------------------------------
   HOW TO ACTIVATE TELEGRAM ALERTS
   ------------------------------------------------------------

   The Telegram code below is already fully wired into the fatigue
   detection logic — nothing else to write. To turn it on:

     1. In Telegram, message @BotFather, send /newbot, follow the
        prompts. It replies with a bot token — copy it.
     2. Message your new bot anything (e.g. "hi") so it can reply to you.
     3. Visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates in a
        browser (with your real token in the URL) and find the number
        after "chat":{"id": — that's your chat ID.
     4. Paste both into TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID below.
     5. Change USE_TELEGRAM from false to true.

   You'll then get a Telegram message the moment fatigueAlert triggers,
   and at most once a minute after that while it stays triggered.

   ============================================================
*/


// LIBRARIES
// ============================================================

#include <Wire.h>

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

#include <ArduinoJson.h>

#include "MAX30105.h"
#include "heartRate.h"

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "secrets.h"  // copy secrets.h.example to secrets.h and fill in your real values


// ============================================================
// PIN DEFINITIONS
// ============================================================

#define SDA_PIN     21
#define SCL_PIN     22

#define LED_PIN     4
#define BUZZER_PIN  5

// AD8232 ECG
#define ECG_PIN       34
#define LO_PLUS_PIN   35
#define LO_MINUS_PIN  32

#define BUTTON_PIN    33

// Status indicator LEDs
#define FINGER_LED_PIN 25   // lights when MAX30102 detects a finger
#define ECG_LED_PIN    26   // lights when ECG leads-off clears (good skin contact)


// ============================================================
// WIFI SETTINGS
// ============================================================

const char* WIFI_SSID     = SECRET_WIFI_SSID;
const char* WIFI_PASSWORD = SECRET_WIFI_PASSWORD;


// ============================================================
// CLOUD SETTINGS
// ============================================================

#define USE_THINGSPEAK true
#define USE_FIREBASE   false
#define USE_TELEGRAM   false   // flip to true once you've filled in the two values below


// ============================================================
// THINGSPEAK
// ============================================================

const char* THINGSPEAK_WRITE_API_KEY = SECRET_THINGSPEAK_WRITE_API_KEY;

const char* THINGSPEAK_URL =
  "https://api.thingspeak.com/update";


// ============================================================
// FIREBASE
// ============================================================

const char* FIREBASE_HOST = SECRET_FIREBASE_HOST;


// ============================================================
// TELEGRAM — see setup instructions in the header comment above
// ============================================================

const char* TELEGRAM_BOT_TOKEN = SECRET_TELEGRAM_BOT_TOKEN;
const char* TELEGRAM_CHAT_ID   = SECRET_TELEGRAM_CHAT_ID;

const unsigned long TELEGRAM_COOLDOWN = 60000;

unsigned long lastTelegramAlert = 0;

bool lastFatigueState = false;


// ============================================================
// SENSOR OBJECTS
// ============================================================

MAX30105 particleSensor;

Adafruit_MPU6050 mpu;


// ============================================================
// MAX30102 HEART RATE VARIABLES
// ============================================================

const byte RATE_SIZE = 4;

byte rates[RATE_SIZE];

byte rateSpot = 0;

long lastBeat = 0;

float beatsPerMinute = 0;

int beatAvg = 0;

unsigned long lastBeatTime = 0;


// ============================================================
// ECG VARIABLES
// ============================================================

int ecgRawValue = 0;
float ecgBaseline = 0;
float ecgFiltered = 0;
int ecgSignalValue = 0;

// Set true for one loop iteration right when a beat is detected —
// used to draw the peak-marker trace in the Serial Plotter
bool ecgPeakJustDetected = false;


// ============================================================
// ECG PEAK DETECTION
// ============================================================
//
// TUNING TIP: watch the Serial Plotter. If the peak marker fires on
// noise between real beats, raise ECG_PEAK_THRESHOLD. If it misses
// visible R-wave spikes, lower it.

const int ECG_PEAK_THRESHOLD = 80;
const int ECG_RESET_THRESHOLD = 25;
const unsigned long ECG_REFRACTORY_MS = 300;

bool ecgAboveThreshold = false;
unsigned long ecgLastPeakTime = 0;


// ============================================================
// ECG RR HISTORY
// ============================================================

const byte RR_HISTORY_SIZE = 8;
unsigned long rrHistory[RR_HISTORY_SIZE];
byte rrIndex = 0;
byte rrCount = 0;


// ============================================================
// FINAL ECG VALUES
// ============================================================

bool ecgLeadsOff = true;
int ecgBpm = 0;
float ecgHrv = 0;
bool ecgMarked = false;


// ============================================================
// TEMPERATURE
// ============================================================

float skinTemp = 0;
unsigned long lastTempRead = 0;
const unsigned long TEMP_READ_INTERVAL = 2000;


// ============================================================
// TIMERS
// ============================================================

unsigned long lastWiFiAttempt = 0;
unsigned long lastSendTime = 0;
unsigned long lastPrintTime = 0;

const unsigned long WIFI_RETRY_INTERVAL = 10000;
const unsigned long SEND_INTERVAL = 20000;


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("=================================");
  Serial.println(" Wearable Fatigue Detection");
  Serial.println(" ESP32 + MAX30102 + MPU6050 + ECG");
  Serial.println("=================================");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  pinMode(FINGER_LED_PIN, OUTPUT);
  pinMode(ECG_LED_PIN, OUTPUT);
  digitalWrite(FINGER_LED_PIN, LOW);
  digitalWrite(ECG_LED_PIN, LOW);

  analogSetPinAttenuation(ECG_PIN, ADC_11db);

  Wire.begin(SDA_PIN, SCL_PIN);

  // ---------------- MAX30102 ----------------
  Serial.println("Initializing MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found!");
    while (true) { blinkError(); }
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);
  Serial.println("MAX30102 detected.");

  // ---------------- MPU6050 ----------------
  Serial.println("Initializing MPU6050...");
  if (!mpu.begin()) {
    Serial.println("ERROR: MPU6050 not found!");
    while (true) { blinkError(); }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("MPU6050 detected.");

  // ---------------- WiFi ----------------
  connectWiFi();

  Serial.println();
  Serial.println("=================================");
  Serial.println(" SETUP COMPLETE");
  Serial.println("=================================");
  Serial.println("Place finger on MAX30102.");
  Serial.println("Wear ECG electrodes.");
  Serial.println("Stay still for accurate ECG.");
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  // ---- 1. MAX30102 heart rate ----
  long irValue = particleSensor.getIR();
  bool fingerDetected = irValue > 50000;

  if (fingerDetected) {
    if (checkForBeat(irValue)) {
      long currentTime = millis();
      long delta = currentTime - lastBeat;
      lastBeat = currentTime;

      if (delta > 250 && delta < 2000) {
        beatsPerMinute = 60.0 / (delta / 1000.0);

        if (beatsPerMinute > 20 && beatsPerMinute < 255) {
          rates[rateSpot] = (byte)beatsPerMinute;
          rateSpot++;
          rateSpot %= RATE_SIZE;

          int total = 0, validRates = 0;
          for (byte i = 0; i < RATE_SIZE; i++) {
            if (rates[i] > 0) { total += rates[i]; validRates++; }
          }
          if (validRates > 0) beatAvg = total / validRates;

          digitalWrite(LED_PIN, HIGH);
          lastBeatTime = millis();
        }
      }
    }
    if (millis() - lastBeatTime > 100) digitalWrite(LED_PIN, LOW);
  } else {
    beatsPerMinute = 0;
    beatAvg = 0;
    digitalWrite(LED_PIN, LOW);
  }

  // ---- 2. MPU6050 ----
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  float motionMag = sqrt(
    accel.acceleration.x * accel.acceleration.x +
    accel.acceleration.y * accel.acceleration.y +
    accel.acceleration.z * accel.acceleration.z
  );

  // ---- 3. Temperature ----
  if (millis() - lastTempRead >= TEMP_READ_INTERVAL) {
    lastTempRead = millis();
    skinTemp = particleSensor.readTemperature();
  }

  // ---- 4. AD8232 ECG ----
  ecgLeadsOff = (digitalRead(LO_PLUS_PIN) == HIGH) || (digitalRead(LO_MINUS_PIN) == HIGH);
  ecgPeakJustDetected = false; // reset every loop; detectEcgPeak sets it true when a beat fires

  digitalWrite(FINGER_LED_PIN, fingerDetected ? HIGH : LOW);
  digitalWrite(ECG_LED_PIN, ecgLeadsOff ? LOW : HIGH);

  if (!ecgLeadsOff) {
    ecgRawValue = analogRead(ECG_PIN);
    detectEcgPeak(ecgRawValue);

    // Always print the plot line — Serial Plotter reads this, Serial
    // Monitor shows it as fast-scrolling numeric text (expected/normal)
    // 4th value = ground-truth button state, so raw capture is pre-labeled
    Serial.print(ecgRawValue);
    Serial.print(",");
    Serial.print((int)ecgBaseline);
    Serial.print(",");
    Serial.print(ecgPeakJustDetected ? (int)ecgBaseline + 300 : (int)ecgBaseline);
    Serial.print(",");
    Serial.println(digitalRead(BUTTON_PIN) == LOW ? 1 : 0);
  } else {
    ecgRawValue = 0;
    ecgSignalValue = 0;
    ecgBaseline = 0;
    ecgFiltered = 0;
    ecgAboveThreshold = false;
    ecgLastPeakTime = 0;
    ecgBpm = 0;
    ecgHrv = 0;

    Serial.print("0,0,0,");
    Serial.println(digitalRead(BUTTON_PIN) == LOW ? 1 : 0);
  }

  // ---- 5. Button marker ----
  ecgMarked = (digitalRead(BUTTON_PIN) == LOW);

  // ---- 6. Fatigue detection (still a placeholder rule) ----
  bool fatigueAlert = false;

  if (fingerDetected && beatAvg > 0 && beatAvg < 55) {
    fatigueAlert = true;
  }
  if (!ecgLeadsOff && ecgHrv > 0 && ecgHrv < 15) {
    fatigueAlert = true;
  }

  digitalWrite(BUZZER_PIN, fatigueAlert ? HIGH : LOW);

  // ---- 7. Readable debug block — always prints once per second too ----
  if (millis() - lastPrintTime >= 1000) {
    lastPrintTime = millis();

    // Machine-readable line for local logging — captures ALL sensors at once.
    // Format: SUMMARY,millis,ppg_bpm,motion_mag,ecg_bpm,ecg_leads_off,skin_temp,
    //         fatigue_alert,ecg_hrv,marked,finger_detected,wifi_connected
    Serial.print("SUMMARY,");
    Serial.print(millis()); Serial.print(",");
    Serial.print(beatAvg); Serial.print(",");
    Serial.print(motionMag, 2); Serial.print(",");
    Serial.print(ecgBpm); Serial.print(",");
    Serial.print(ecgLeadsOff ? 1 : 0); Serial.print(",");
    Serial.print(skinTemp, 1); Serial.print(",");
    Serial.print(fatigueAlert ? 1 : 0); Serial.print(",");
    Serial.print(ecgHrv, 1); Serial.print(",");
    Serial.print(ecgMarked ? 1 : 0); Serial.print(",");
    Serial.print(fingerDetected ? 1 : 0); Serial.print(",");
    Serial.println(WiFi.status() == WL_CONNECTED ? 1 : 0);

    Serial.println("--------------------------------");
    Serial.print("IR Value: "); Serial.println(irValue);
    Serial.print("Finger: "); Serial.println(fingerDetected ? "YES" : "NO");
    Serial.print("PPG Average BPM: "); Serial.println(beatAvg);
    Serial.print("Skin Temp: "); Serial.print(skinTemp); Serial.println(" C");
    Serial.print("Motion Magnitude: "); Serial.println(motionMag);

    Serial.print("ECG Leads Off: "); Serial.print(ecgLeadsOff ? "YES" : "NO");
    Serial.print(" (LO+: "); Serial.print(digitalRead(LO_PLUS_PIN));
    Serial.print(", LO-: "); Serial.print(digitalRead(LO_MINUS_PIN)); Serial.print(")");
    Serial.print(" | Raw ECG: "); Serial.print(ecgRawValue);
    Serial.print(" | Baseline: "); Serial.print((int)ecgBaseline);
    Serial.print(" | Signal: "); Serial.print(ecgSignalValue);
    Serial.print(" | ECG BPM: "); Serial.print(ecgBpm);
    Serial.print(" | ECG HRV: "); Serial.print(ecgHrv);
    Serial.print(" | Marked: "); Serial.println(ecgMarked ? "YES" : "NO");

    Serial.print("Fatigue Alert: "); Serial.println(fatigueAlert ? "YES" : "NO");
    Serial.print("WiFi: "); Serial.println(WiFi.status() == WL_CONNECTED ? "CONNECTED" : "DISCONNECTED");
  }

  // ---- 8. WiFi reconnection ----
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastWiFiAttempt >= WIFI_RETRY_INTERVAL) {
      lastWiFiAttempt = millis();
      connectWiFi();
    }
  }

  // ---- 9. Cloud uploads ----
  if (millis() - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = millis();

    if (USE_THINGSPEAK) {
      sendThingSpeak(beatAvg, motionMag, fingerDetected, fatigueAlert);
    }
    if (USE_FIREBASE) {
      sendFirebase(irValue, beatAvg, fingerDetected, accel, gyro, fatigueAlert);
    }
  }

  // ---- 10. Telegram alert ----
  if (USE_TELEGRAM) {
    bool justTriggered = fatigueAlert && !lastFatigueState;
    bool cooldownExpired = fatigueAlert && (millis() - lastTelegramAlert >= TELEGRAM_COOLDOWN);

    if (justTriggered || cooldownExpired) {
      lastTelegramAlert = millis();
      String msg = "Fatigue alert! PPG BPM: " + String(beatAvg) +
                   ", ECG BPM: " + String(ecgBpm) +
                   ", HRV: " + String(ecgHrv);
      sendTelegramAlert(msg);
    }
    lastFatigueState = fatigueAlert;
  }

  // 4 ms ≈ 250 samples/second — matches typical AD8232 tutorials
  delay(4);
}


// ============================================================
// ECG R-PEAK DETECTION
// ============================================================

void detectEcgPeak(int sample) {

  unsigned long now = millis();

  if (ecgBaseline == 0) {
    ecgBaseline = sample; // first sample after reconnecting — seed the baseline
  }

  ecgBaseline = (ecgBaseline * 0.995) + (sample * 0.005);

  ecgFiltered = sample - ecgBaseline;
  ecgSignalValue = abs((int)ecgFiltered);

  if (ecgSignalValue > ECG_PEAK_THRESHOLD && !ecgAboveThreshold) {
    ecgAboveThreshold = true;

    if (now - ecgLastPeakTime > ECG_REFRACTORY_MS) {
      if (ecgLastPeakTime > 0) {
        unsigned long rr = now - ecgLastPeakTime;

        if (rr > 300 && rr < 2000) { // 30-200 BPM plausible range
          ecgBpm = 60000 / rr;
          recordRR(rr);
          ecgPeakJustDetected = true;
        }
      }
      ecgLastPeakTime = now;
    }
  }

  if (ecgSignalValue < ECG_RESET_THRESHOLD) {
    ecgAboveThreshold = false;
  }
}


// ============================================================
// RECORD RR + CALCULATE HRV
// ============================================================

void recordRR(unsigned long rr) {
  rrHistory[rrIndex] = rr;
  rrIndex++;
  rrIndex %= RR_HISTORY_SIZE;
  if (rrCount < RR_HISTORY_SIZE) rrCount++;

  if (rrCount >= 2) {
    float sumDiff = 0;
    int pairs = 0;

    for (int i = 1; i < rrCount; i++) {
      int idxA = (rrIndex + RR_HISTORY_SIZE - i) % RR_HISTORY_SIZE;
      int idxB = (rrIndex + RR_HISTORY_SIZE - i - 1) % RR_HISTORY_SIZE;
      long diff = (long)rrHistory[idxA] - (long)rrHistory[idxB];
      sumDiff += abs(diff);
      pairs++;
    }

    if (pairs > 0) ecgHrv = sumDiff / pairs;
  }
}


// ============================================================
// WIFI CONNECTION
// ============================================================

void connectWiFi() {
  Serial.println();
  Serial.println("Connecting to WiFi...");

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(500);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startTime = millis();
  const unsigned long timeout = 15000;

  while (WiFi.status() != WL_CONNECTED && millis() - startTime < timeout) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi CONNECTED!");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi connection FAILED.");
  }
}


// ============================================================
// THINGSPEAK UPLOAD
// ============================================================

void sendThingSpeak(int bpm, float motionMag, bool fingerDetected, bool fatigueAlert) {
  if (WiFi.status() != WL_CONNECTED) return;

  String url = String(THINGSPEAK_URL) + "?api_key=" + THINGSPEAK_WRITE_API_KEY;
  url += "&field1=" + String(bpm);
  url += "&field2=" + String(motionMag, 2);
  url += "&field3=" + String(ecgBpm);
  url += "&field4=" + String(ecgLeadsOff ? 1 : 0);
  url += "&field5=" + String(skinTemp, 1);
  url += "&field6=" + String(fatigueAlert ? 1 : 0);
  url += "&field7=" + String(ecgHrv, 1);
  url += "&field8=" + String(ecgMarked ? 1 : 0); // ground-truth label for ML training

  HTTPClient http;
  http.begin(url);
  int httpCode = http.GET();
  Serial.print("ThingSpeak response code: ");
  Serial.println(httpCode);
  http.end();
}


// ============================================================
// FIREBASE UPLOAD
// ============================================================

void sendFirebase(long irValue, int bpm, bool fingerDetected,
                   sensors_event_t &accel, sensors_event_t &gyro, bool fatigueAlert) {
  if (WiFi.status() != WL_CONNECTED) return;

  StaticJsonDocument<512> doc;
  doc["timestamp"] = millis();
  doc["bpm"] = bpm;
  doc["ir_value"] = irValue;
  doc["finger_detected"] = fingerDetected;
  doc["skin_temp"] = skinTemp;
  doc["ecg_bpm"] = ecgBpm;
  doc["ecg_hrv"] = ecgHrv;
  doc["ecg_marked"] = ecgMarked; // ground-truth label for ML training
  doc["fatigue_alert"] = fatigueAlert;

  String payload;
  serializeJson(doc, payload);

  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  String url = "https://" + String(FIREBASE_HOST) + "/readings.json";
  http.begin(client, url);
  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST(payload);
  Serial.print("Firebase response code: ");
  Serial.println(httpCode);
  http.end();
}


// ============================================================
// TELEGRAM ALERT
// ============================================================

void sendTelegramAlert(String message) {
  if (WiFi.status() != WL_CONNECTED) return;

  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  String url = "https://api.telegram.org/bot" + String(TELEGRAM_BOT_TOKEN) +
               "/sendMessage?chat_id=" + String(TELEGRAM_CHAT_ID) +
               "&text=" + urlEncode(message);

  http.begin(client, url);
  int httpCode = http.GET();
  Serial.print("Telegram response code: ");
  Serial.println(httpCode);
  http.end();
}


// ============================================================
// URL ENCODE
// ============================================================

String urlEncode(String str) {
  String encoded = "";
  for (int i = 0; i < str.length(); i++) {
    char c = str.charAt(i);
    if (isalnum(c)) {
      encoded += c;
    } else if (c == ' ') {
      encoded += "%20";
    } else {
      char code0 = "0123456789ABCDEF"[(c >> 4) & 0xF];
      char code1 = "0123456789ABCDEF"[c & 0xF];
      encoded += '%';
      encoded += code0;
      encoded += code1;
    }
  }
  return encoded;
}


// ============================================================
// ERROR LED
// ============================================================

void blinkError() {
  digitalWrite(LED_PIN, HIGH); delay(200);
  digitalWrite(LED_PIN, LOW);  delay(200);
}
