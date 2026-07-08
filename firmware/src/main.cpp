#include <Arduino.h>

#ifndef NOSEKNOWS_SAMPLE_PERIOD_MS
#define NOSEKNOWS_SAMPLE_PERIOD_MS 100
#endif

// ESP32-S3 ADC1 pins. Keep the first hardware pass on ADC1 to avoid ADC2/Wi-Fi
// conflicts and board-specific strapping surprises.
static constexpr uint8_t SENSOR_PINS[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
static constexpr size_t SENSOR_COUNT = sizeof(SENSOR_PINS) / sizeof(SENSOR_PINS[0]);

static uint32_t sequence_number = 0;
static uint32_t last_sample_ms = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  for (size_t i = 0; i < SENSOR_COUNT; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
    analogSetPinAttenuation(SENSOR_PINS[i], ADC_11db);
  }

  Serial.println("NK_BOOT,board=esp32-s3,channels=9,resolution_bits=12");
  Serial.print("NK_PINS");
  for (size_t i = 0; i < SENSOR_COUNT; i++) {
    Serial.print(",gpio");
    Serial.print(i);
    Serial.print("=");
    Serial.print(SENSOR_PINS[i]);
  }
  Serial.println();
  Serial.println("NK_HEADER,seq,ms,adc0,adc1,adc2,adc3,adc4,adc5,adc6,adc7,adc8");
}

void loop() {
  const uint32_t now_ms = millis();
  if (now_ms - last_sample_ms < NOSEKNOWS_SAMPLE_PERIOD_MS) {
    return;
  }
  last_sample_ms = now_ms;

  uint16_t readings[SENSOR_COUNT];
  for (size_t i = 0; i < SENSOR_COUNT; i++) {
    readings[i] = analogRead(SENSOR_PINS[i]);
  }

  Serial.print("NK_ADC,");
  Serial.print(sequence_number++);
  Serial.print(",");
  Serial.print(now_ms);
  for (size_t i = 0; i < SENSOR_COUNT; i++) {
    Serial.print(",");
    Serial.print(readings[i]);
  }
  Serial.println();
}

