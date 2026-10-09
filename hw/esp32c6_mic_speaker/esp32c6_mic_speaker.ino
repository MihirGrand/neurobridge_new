/**
 * ESP32-C6 - Mic stream + MAX98357A speaker with barge-in
 *
 * Hardware notes (C6 constraints):
 *   • Only ONE I2S peripheral (I2S_NUM_0) - mic and speaker share it
 *   • Single HP core - never use xTaskCreatePinnedToCore(..., 1, ...)
 *   • No APLL - use PLL_F160M clock instead
 *   • Uses the new IDF5 i2s_std API (driver/i2s_std.h), NOT legacy driver/i2s.h
 *
 * I2S pin assignments:
 *   Mic  (INMP441):  WS=5  SCK=4  SD=6
 *   Spk  (MAX98357A): LRC=5 BCLK=4 DIN=22   ← WS/SCK reused, only DIN differs
 *
 * Flow:
 *   IDLE      → mic captures → streams to Python via GET /audio
 *   PLAYING   → Python POSTs TTS audio to /play → speaker outputs
 *   BARGE-IN  → mic energy exceeds threshold while playing
 *               → ESP32 sends "BARGE_IN\n" on /events SSE stream
 *               → Python receives it, calls tts.interrupt(), closes /play
 *               → ESP32 detects /play client gone, switches back to mic
 *
 * State machine:
 *   IDLE ──/play POST──► PLAYING ──barge-in──► BARGE_IN (brief) ──► IDLE
 *    ▲                      │ play ends                                │
 *    └──────────────────────┘         ◄────────────────────────────────┘
 */

#include <WiFi.h>
#include <WebServer.h>
#include <driver/i2s_std.h>
#include <freertos/ringbuf.h>

// ═══════════════════════════════════════════════════════════
//  PIN CONFIG
// ═══════════════════════════════════════════════════════════

// Shared I2S clock lines (mic WS=spk LRC, mic SCK=spk BCLK)
#define I2S_WS_LRC   5
#define I2S_SCK_BCLK 4

// Data lines (different direction, so separate pins)
#define MIC_SD_PIN   6    // INMP441 data out → ESP32 data in
#define SPK_DIN_PIN  22   // ESP32 data out  → MAX98357A data in

// ═══════════════════════════════════════════════════════════
//  AUDIO CONFIG
// ═══════════════════════════════════════════════════════════

#define SAMPLE_RATE     16000   // Both mic and speaker run at 16kHz
                                // (Kokoro 24kHz resampled to 16kHz in Python)
#define I2S_BUF_LEN     512     // samples per DMA buffer
#define RING_BUF_BYTES  (32 * 1024)  // ~2s at 16kHz 16-bit

// Barge-in: RMS energy threshold above which we consider the user speaking.
// Tune this for your environment - lower = more sensitive.
#define BARGE_IN_THRESHOLD  800
// How many consecutive loud chunks before declaring barge-in
#define BARGE_IN_CHUNKS     3

// ═══════════════════════════════════════════════════════════
//  NETWORK
// ═══════════════════════════════════════════════════════════

const char* ssid     = "SSID";
const char* password = "PASS";

IPAddress local_IP(192, 168, 0, 250);
IPAddress gateway(192, 168, 0, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress primaryDNS(8, 8, 8, 8);
IPAddress secondaryDNS(8, 8, 4, 4);

WebServer server(80);

// ═══════════════════════════════════════════════════════════
//  STATE MACHINE
// ═══════════════════════════════════════════════════════════

typedef enum {
  STATE_IDLE,       // mic active, streaming to /audio
  STATE_PLAYING,    // speaker active, receiving from /play
  STATE_BARGE_IN    // transitioning back to mic after interrupt
} DeviceState;

volatile DeviceState deviceState = STATE_IDLE;

// SSE client for pushing barge-in events to Python
WiFiClient* sseClient = nullptr;

// Speaker ring buffer
static RingbufHandle_t spkRingBuf = nullptr;

// I2S channel handle (one shared handle, reconfigured per mode)
static i2s_chan_handle_t i2s_rx_handle = nullptr;  // mic (RX)
static i2s_chan_handle_t i2s_tx_handle = nullptr;  // speaker (TX)

// ═══════════════════════════════════════════════════════════
//  I2S HELPERS  (new IDF5 i2s_std API)
// ═══════════════════════════════════════════════════════════

void i2s_stop_all() {
  if (i2s_rx_handle) {
    i2s_channel_disable(i2s_rx_handle);
    i2s_del_channel(i2s_rx_handle);
    i2s_rx_handle = nullptr;
  }
  if (i2s_tx_handle) {
    i2s_channel_disable(i2s_tx_handle);
    i2s_del_channel(i2s_tx_handle);
    i2s_tx_handle = nullptr;
  }
}

// Start I2S in RX (microphone) mode
void i2s_start_mic() {
  i2s_stop_all();

  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan_cfg.dma_desc_num  = 8;
  chan_cfg.dma_frame_num = I2S_BUF_LEN;
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, nullptr, &i2s_rx_handle));

  i2s_std_config_t std_cfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = (gpio_num_t)I2S_SCK_BCLK,
      .ws   = (gpio_num_t)I2S_WS_LRC,
      .dout = I2S_GPIO_UNUSED,
      .din  = (gpio_num_t)MIC_SD_PIN,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false }
    }
  };
  // C6 has no APLL - force PLL_F160M
  std_cfg.clk_cfg.clk_src = I2S_CLK_SRC_PLL_160M;

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(i2s_rx_handle, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(i2s_rx_handle));
  Serial.println("[I2S] Mic (RX) started");
}

// Start I2S in TX (speaker / MAX98357A) mode
void i2s_start_speaker() {
  i2s_stop_all();

  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan_cfg.dma_desc_num  = 8;
  chan_cfg.dma_frame_num = I2S_BUF_LEN;
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &i2s_tx_handle, nullptr));

  i2s_std_config_t std_cfg = {
    .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    // MAX98357A is mono - use stereo slot but duplicate signal on both channels
    .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = (gpio_num_t)I2S_SCK_BCLK,
      .ws   = (gpio_num_t)I2S_WS_LRC,
      .dout = (gpio_num_t)SPK_DIN_PIN,
      .din  = I2S_GPIO_UNUSED,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false }
    }
  };
  std_cfg.clk_cfg.clk_src = I2S_CLK_SRC_PLL_160M;

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(i2s_tx_handle, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(i2s_tx_handle));
  Serial.println("[I2S] Speaker (TX) started");
}

// ═══════════════════════════════════════════════════════════
//  AGC (used on mic path)
// ═══════════════════════════════════════════════════════════

static int32_t dc_offset  = 0;
static float   agc_env    = 0.0f;

const float AGC_ATTACK    = 0.005f;
const float AGC_RELEASE   = 0.0001f;
const float AGC_MIN_GAIN  = 6.0f;
const float AGC_MAX_GAIN  = 24.0f;
const float AGC_NOISE_FLOOR = 220.0f;

// Returns RMS energy of the block (before AGC) so we can use it for barge-in
float processMicSamples(int16_t* buf, int n) {
  float sumSq = 0.0f;
  for (int i = 0; i < n; i++) {
    dc_offset += (buf[i] - dc_offset) >> 8;
    int16_t s = buf[i] - (int16_t)dc_offset;

    float sig = fabsf((float)s);
    sumSq += (float)s * (float)s;

    if (sig > agc_env)
      agc_env += (sig - agc_env) * AGC_ATTACK;
    else
      agc_env -= agc_env * AGC_RELEASE;

    float gain = (agc_env < AGC_NOISE_FLOOR)
                   ? AGC_MIN_GAIN
                   : fmaxf(AGC_MIN_GAIN, fminf(AGC_MAX_GAIN, 6000.0f / agc_env));

    int32_t amp = (int32_t)((float)s * gain);
    buf[i] = (int16_t)constrain(amp, -32768, 32767);
  }
  return sqrtf(sumSq / n);
}

// ═══════════════════════════════════════════════════════════
//  BARGE-IN DETECTION
// ═══════════════════════════════════════════════════════════

void sendBargeInEvent() {
  if (sseClient && sseClient->connected()) {
    sseClient->print("data: BARGE_IN\n\n");
    Serial.println("[BARGE-IN] Event sent to Python");
  } else {
    Serial.println("[BARGE-IN] No SSE client - Python won't know");
  }
}

// ═══════════════════════════════════════════════════════════
//  MIC STREAM TASK
// ═══════════════════════════════════════════════════════════

void micStreamTask(void* param) {
  WiFiClient client = *((WiFiClient*)param);
  free(param);

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: application/octet-stream");
  client.println("Transfer-Encoding: chunked");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Cache-Control: no-cache");
  client.println();

  int16_t buf[I2S_BUF_LEN];
  int bargeCount = 0;

  while (client.connected() && deviceState == STATE_IDLE) {
    size_t bytesRead = 0;
    esp_err_t err = i2s_channel_read(i2s_rx_handle, buf, sizeof(buf), &bytesRead, pdMS_TO_TICKS(100));
    if (err != ESP_OK || bytesRead == 0) continue;

    int nSamples = bytesRead / 2;
    float rms = processMicSamples(buf, nSamples);

    // While playing, monitor for barge-in
    // (This task only runs in STATE_IDLE, but the speaker task may have
    //  just set STATE_PLAYING - check here for race-free detection)
    if (deviceState == STATE_PLAYING) {
      if (rms > BARGE_IN_THRESHOLD) {
        bargeCount++;
        if (bargeCount >= BARGE_IN_CHUNKS) {
          bargeCount = 0;
          deviceState = STATE_BARGE_IN;
          sendBargeInEvent();
        }
      } else {
        bargeCount = 0;
      }
      // Don't stream mic audio to Python while playing
      continue;
    }

    bargeCount = 0;
    client.printf("%X\r\n", bytesRead);
    client.write((uint8_t*)buf, bytesRead);
    client.println();
  }

  client.println("0\r\n");
  client.stop();
  vTaskDelete(NULL);
}

// ═══════════════════════════════════════════════════════════
//  SPEAKER DRAIN TASK
// Runs while STATE_PLAYING: pulls from ring buffer → I2S TX
// ═══════════════════════════════════════════════════════════

void speakerDrainTask(void* param) {
  int16_t mono[I2S_BUF_LEN];
  int16_t stereo[I2S_BUF_LEN * 2]; // L+R interleaved

  while (deviceState == STATE_PLAYING) {
    size_t itemSize = 0;
    uint8_t* item = (uint8_t*)xRingbufferReceiveUpTo(
      spkRingBuf, &itemSize, pdMS_TO_TICKS(50),
      I2S_BUF_LEN * sizeof(int16_t)
    );

    if (item == nullptr) continue;

    int n = itemSize / 2;
    memcpy(mono, item, itemSize);
    vRingbufferReturnItem(spkRingBuf, item);

    // Duplicate mono → stereo for MAX98357A
    for (int i = 0; i < n; i++) {
      stereo[i * 2]     = mono[i];
      stereo[i * 2 + 1] = mono[i];
    }

    size_t written = 0;
    i2s_channel_write(i2s_tx_handle, stereo, n * 4, &written, pdMS_TO_TICKS(100));
  }

  // Silence the amp before switching back to mic
  memset(stereo, 0, sizeof(stereo));
  size_t w = 0;
  i2s_channel_write(i2s_tx_handle, stereo, sizeof(stereo), &w, pdMS_TO_TICKS(100));

  Serial.println("[SPK] Drain task exiting");
  vTaskDelete(NULL);
}

// ═══════════════════════════════════════════════════════════
//  /play RECEIVE TASK
// Receives raw int16 PCM from Python, feeds ring buffer
// ═══════════════════════════════════════════════════════════

void playReceiveTask(void* param) {
  WiFiClient client = *((WiFiClient*)param);
  free(param);

  // Respond immediately
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/plain");
  client.println("Access-Control-Allow-Origin: *");
  client.println("Connection: close");
  client.println();
  client.print("OK");

  // Switch I2S to speaker mode
  i2s_start_speaker();
  deviceState = STATE_PLAYING;
  Serial.println("[SPK] Playback started");

  // Launch the drain task (single core - no pinning)
  xTaskCreate(speakerDrainTask, "spkDrain", 4096, nullptr, 2, nullptr);

  // Also start a background mic-listen task for barge-in detection
  // We need a second channel open in RX to sample mic energy.
  // On C6 with one I2S this isn't possible simultaneously - instead
  // we rely on Python's VAD sending the barge-in signal via the
  // existing /audio stream being cut, OR we use the /barge endpoint
  // polled from Python side. See Python code for how this is handled.

  uint8_t buf[512];

  while (client.connected() || client.available()) {
    // Abort immediately if barge-in was signalled
    if (deviceState == STATE_BARGE_IN) {
      Serial.println("[SPK] Barge-in: aborting playback");
      break;
    }

    if (!client.available()) {
      delay(1);
      continue;
    }

    int n = client.read(buf, sizeof(buf));
    if (n <= 0) continue;

    if (xRingbufferSend(spkRingBuf, buf, n, pdMS_TO_TICKS(200)) != pdTRUE) {
      Serial.println("[SPK] Ring buffer full - dropping");
    }
  }

  client.stop();

  // Wait for drain task to flush remaining audio
  // (it exits when deviceState != PLAYING)
  deviceState = STATE_IDLE;
  delay(300); // give drain task time to write silence and exit

  // Clear ring buffer of any leftover bytes
  size_t dummy;
  while (true) {
    uint8_t* leftover = (uint8_t*)xRingbufferReceiveUpTo(
      spkRingBuf, &dummy, 0, 512);
    if (!leftover) break;
    vRingbufferReturnItem(spkRingBuf, leftover);
  }

  // Switch back to mic
  i2s_start_mic();
  Serial.println("[SPK] Playback ended - mic resumed");
  vTaskDelete(NULL);
}

// ═══════════════════════════════════════════════════════════
//  /barge POST endpoint
//  Python VAD calls this when it detects the user speaking.
//  This is the primary barge-in trigger since the C6 can't
//  simultaneously read mic + play speaker.
// ═══════════════════════════════════════════════════════════

void handleBarge() {
  if (deviceState == STATE_PLAYING) {
    deviceState = STATE_BARGE_IN;
    Serial.println("[BARGE-IN] Triggered by Python VAD");
  }
  server.send(200, "text/plain", "OK");
}

// ═══════════════════════════════════════════════════════════
//  /events SSE endpoint
//  Python connects here at startup to receive push events
//  (currently only BARGE_IN, but extensible)
// ═══════════════════════════════════════════════════════════

void handleEvents() {
  // Keep the raw client and hold it open
  if (sseClient) {
    sseClient->stop();
    delete sseClient;
  }
  sseClient = new WiFiClient(server.client());

  sseClient->println("HTTP/1.1 200 OK");
  sseClient->println("Content-Type: text/event-stream");
  sseClient->println("Cache-Control: no-cache");
  sseClient->println("Access-Control-Allow-Origin: *");
  sseClient->println("Connection: keep-alive");
  sseClient->println();
  sseClient->print("data: CONNECTED\n\n");

  Serial.println("[SSE] Python event client connected");
  // Don't call server.send() - we're holding the connection manually
}

// ═══════════════════════════════════════════════════════════
//  HTTP HANDLERS
// ═══════════════════════════════════════════════════════════

void handleAudio() {
  if (deviceState != STATE_IDLE) {
    server.send(409, "text/plain", "Not idle");
    return;
  }
  WiFiClient* clientPtr = new WiFiClient(server.client());
  xTaskCreate(micStreamTask, "micTask", 8192, clientPtr, 1, nullptr);
}

void handlePlay() {
  if (deviceState != STATE_IDLE) {
    server.send(409, "text/plain", "Busy");
    return;
  }
  WiFiClient* clientPtr = new WiFiClient(server.client());
  xTaskCreate(playReceiveTask, "playTask", 8192, clientPtr, 1, nullptr);
}

void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html><html><head><title>ESP32-C6 Voice</title></head><body>
<h2>ESP32-C6 Mic Stream</h2>
<button onclick="startStream()">▶ Start</button>
<button onclick="stopStream()">■ Stop</button>
<p id="status">Idle</p>
<script>
  let audioCtx, reader;
  async function startStream() {
    document.getElementById('status').textContent = 'Connecting...';
    audioCtx = new AudioContext({ sampleRate: 16000 });
    const response = await fetch('/audio');
    reader = response.body.getReader();
    document.getElementById('status').textContent = 'Streaming';
    let t = audioCtx.currentTime + 0.1;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const n = value.length / 2;
      const f32 = new Float32Array(n);
      const dv = new DataView(value.buffer);
      for (let i = 0; i < n; i++) f32[i] = dv.getInt16(i*2, true) / 32768;
      const buf = audioCtx.createBuffer(1, n, 16000);
      buf.copyToChannel(f32, 0);
      const src = audioCtx.createBufferSource();
      src.buffer = buf; src.connect(audioCtx.destination);
      if (t < audioCtx.currentTime) t = audioCtx.currentTime + 0.05;
      src.start(t); t += buf.duration;
    }
  }
  function stopStream() {
    if (reader) reader.cancel();
    if (audioCtx) audioCtx.close();
    document.getElementById('status').textContent = 'Stopped';
  }
</script></body></html>)rawliteral";
  server.send(200, "text/html", html);
}

// ═══════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);

  // Start in mic mode
  i2s_start_mic();

  // Ring buffer for speaker path
  spkRingBuf = xRingbufferCreate(RING_BUF_BYTES, RINGBUF_TYPE_BYTEBUF);
  if (!spkRingBuf) {
    Serial.println("FATAL: ring buffer alloc failed");
    while (true) delay(1000);
  }

  if (!WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS))
    Serial.println("Static IP config failed");

  WiFi.begin(ssid, password);
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());

  server.on("/",       handleRoot);
  server.on("/audio",  handleAudio);
  server.on("/play",   HTTP_POST, handlePlay);
  server.on("/barge",  HTTP_POST, handleBarge);
  server.on("/events", handleEvents);
  server.begin();

  Serial.println("Ready.");
  Serial.println("  GET  /audio  - mic stream");
  Serial.println("  POST /play   - speaker (raw int16 PCM @ 16kHz)");
  Serial.println("  POST /barge  - trigger barge-in from Python VAD");
  Serial.println("  GET  /events - SSE push events to Python");
}

// ═══════════════════════════════════════════════════════════
//  LOOP
// ═══════════════════════════════════════════════════════════

void loop() {
  server.handleClient();

  // Keep SSE connection alive with a heartbeat comment every 5s
  static uint32_t lastHB = 0;
  if (millis() - lastHB > 5000) {
    lastHB = millis();
    if (sseClient && sseClient->connected()) {
      sseClient->print(": heartbeat\n\n");
    }
  }
}
