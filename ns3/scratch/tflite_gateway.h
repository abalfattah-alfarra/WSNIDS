// tflite_gateway.h
#pragma once
#include <string>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

class TfliteGateway
{
public:
    explicit TfliteGateway(const std::string& modelPath, uint32_t batch=32);
    void   AddWindow(const std::vector<float>& win);
    void   Flush();   // force inference
private:
    void   RunInference();
    uint32_t               m_batch, m_dim;
    std::unique_ptr<tflite::Interpreter> m_interp;
    std::vector<std::vector<float>>      m_queue;
};
