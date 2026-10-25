// SPDX-License-Identifier: Apache-2.0
// Ordered specialization of Candle's conv_transpose1d, fork 2266099d.
// Original source: candle-kernels/src/conv.cu (Apache-2.0 OR MIT).
// See SOURCE.md and LICENSE.candle-APACHE / LICENSE.candle-MIT.
#include <cuda_runtime.h>

template<int Cin, int Cout, int Stride>
__device__ void ordered_body(float* out, const float* x, const float* w,
                             int len, unsigned count) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    constexpr int Padding = (Stride + 1) / 2;
    constexpr int Kernel = 2 * Stride;
    const int out_len = len * Stride;
    const int ox = id % out_len;
    const int oc = (id / out_len) % Cout;
    const int batch = id / (out_len * Cout);
    const int first_tap = (ox + Padding) % Stride;
    float sum = 0.0f;
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int tap = first_tap + part * Stride;
        const int ix = (ox + Padding - tap) / Stride;
        if (ix < 0 || ix >= len) continue;
        // Preserve tap -> channel order and each individual F32 rounding.
        #pragma unroll 4
        for (int ic = 0; ic < Cin; ++ic) {
            sum = __fmaf_rn(x[(batch * Cin + ic) * len + ix],
                            w[(ic * Cout + oc) * Kernel + tap], sum);
        }
    }
    out[id] = sum;
}
#define DECONV(CI, CO, S) \
extern "C" __global__ void ordered_deconv_##CI##_##CO##_##S( \
float* out, const float* x, const float* w, int len, unsigned count) { \
ordered_body<CI, CO, S>(out, x, w, len, count); }
DECONV(1024, 512, 8)
DECONV(512, 256, 5)
DECONV(256, 128, 4)
DECONV(128, 64, 2)
DECONV(64, 32, 3)
