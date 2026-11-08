#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#define QMINI_EXPORT extern "C" __declspec(dllexport)
#else
#define QMINI_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Four packed 2-bit codes per output row: 00=-2, 01=-1, 10=+1, 11=+2.
// The accumulator uses integer add/subtract/doubling; scales are applied once
// after accumulation, avoiding a floating-point multiply in the inner loop.
QMINI_EXPORT int qmini_linear_i8(
    const int8_t* input,
    const uint8_t* packed_weights,
    const float* weight_scales,
    const float* input_scales,
    const float* bias,
    float* output,
    int rows,
    int in_features,
    int out_features
) {
    if (!input || !packed_weights || !weight_scales || !input_scales || !output) {
        return 1;
    }
    if (rows <= 0 || in_features <= 0 || out_features <= 0 ||
        in_features > std::numeric_limits<int>::max() - 3) {
        return 2;
    }

    for (int row = 0; row < rows; ++row) {
        if (!std::isfinite(input_scales[row]) || input_scales[row] <= 0.0f) {
            return 3;
        }
    }
    for (int output_index = 0; output_index < out_features; ++output_index) {
        if (!std::isfinite(weight_scales[output_index]) ||
            weight_scales[output_index] <= 0.0f) {
            return 4;
        }
        if (bias && !std::isfinite(bias[output_index])) {
            return 5;
        }
    }

    const int bytes_per_row = (in_features + 3) / 4;
    for (int row = 0; row < rows; ++row) {
        for (int output_index = 0; output_index < out_features; ++output_index) {
            int64_t accumulator = 0;
            const uint8_t* weight_row =
                packed_weights + static_cast<std::size_t>(output_index) *
                static_cast<std::size_t>(bytes_per_row);
            const int8_t* input_row =
                input + static_cast<std::size_t>(row) *
                static_cast<std::size_t>(in_features);

            for (int feature = 0; feature < in_features; ++feature) {
                const uint8_t packed = weight_row[feature / 4];
                const uint8_t code = static_cast<uint8_t>(
                    (packed >> ((feature % 4) * 2)) & 0x03
                );
                const int32_t value = static_cast<int32_t>(input_row[feature]);
                int32_t term = 0;
                switch (code) {
                    case 0: term = -(value + value); break; // -2
                    case 1: term = -value; break;           // -1
                    case 2: term = value; break;            // +1
                    case 3: term = value + value; break;    // +2
                    default: return 6;
                }
                accumulator += static_cast<int64_t>(term);
            }

            const float offset = bias ? bias[output_index] : 0.0f;
            const float scaled = static_cast<float>(accumulator) *
                input_scales[row] * weight_scales[output_index];
            output[static_cast<std::size_t>(row) *
                static_cast<std::size_t>(out_features) +
                static_cast<std::size_t>(output_index)] = scaled + offset;
        }
    }
    return 0;
}
