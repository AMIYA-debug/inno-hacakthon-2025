#include "model_data.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Allocate memory for the TensorFlow Lite model
constexpr int kTensorArenaSize = 100 * 1024;  // adjust if model is large
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing TensorFlow Lite model...");

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model ready!");
}

void loop() {
  // 🔹 Replace this with your camera input
  // Example: fill input tensor with dummy values
  for (int i = 0; i < input->bytes; i++) {
    input->data.uint8[i] = random(0, 255);
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // 🔹 Read output
  float prediction = output->data.f[0];  // example: first output
  Serial.print("Prediction: ");
  Serial.println(prediction);

  delay(2000);
}
