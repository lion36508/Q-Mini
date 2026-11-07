#include <cstdint>
#include <cstddef>
#if defined(_WIN32)
#define QMINI_EXPORT extern "C" __declspec(dllexport)
#else
#define QMINI_EXPORT extern "C" __attribute__((visibility("default")))
#endif
// Packed layout: four 2-bit values per byte: 00=-2, 01=-1, 10=+1, 11=+2.
QMINI_EXPORT int qmini_linear_i8(const int8_t* input, const uint8_t* packed_weights,
    const float* weight_scales, const float* input_scales, const float* bias,
    float* output, int rows, int in_features, int out_features) {
    if (!input || !packed_weights || !weight_scales || !input_scales || !output ||
        rows <= 0 || in_features <= 0 || out_features <= 0) return 1;
    const int bytes_per_row = (in_features + 3) / 4;
    for (int r = 0; r < rows; ++r) for (int o = 0; o < out_features; ++o) {
        int32_t acc = 0;
        const uint8_t* w = packed_weights + static_cast<std::size_t>(o) * bytes_per_row;
        for (int i = 0; i < in_features; ++i) {
            uint8_t code = (w[i / 4] >> ((i % 4) * 2)) & 3;
            int32_t q = code == 0 ? -2 : code == 1 ? -1 : code == 2 ? 1 : 2;
            acc += static_cast<int32_t>(input[static_cast<std::size_t>(r) * in_features + i]) * q;
        }
        float b = bias ? bias[o] : 0.0f;
        output[static_cast<std::size_t>(r) * out_features + o] =
            static_cast<float>(acc) * input_scales[r] * weight_scales[o] + b;
    }
    return 0;
}
