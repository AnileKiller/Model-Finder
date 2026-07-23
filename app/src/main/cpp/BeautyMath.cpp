#include "BeautyMath.h"

#include <cstring>   // std::memcpy
#include <vector>

namespace beautymath {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Channel extraction from packed ARGB.
inline int chA(int p) { return static_cast<int>(static_cast<uint32_t>(p) >> 24); }
inline int chR(int p) { return (p >> 16) & 0xFF; }
inline int chG(int p) { return (p >> 8)  & 0xFF; }
inline int chB(int p) { return  p        & 0xFF; }

inline int packOpaque(int r, int g, int b) {
    return static_cast<int>(0xFF000000u) | (r << 16) | (g << 8) | b;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Pixel math
// ---------------------------------------------------------------------------

int blendPixel(int a, int b, float alpha) {
    const float t = clampf(alpha, 0.0f, 1.0f);

    const int ar = chR(a), br = chR(b);
    const int ag = chG(a), bg = chG(b);
    const int ab = chB(a), bb = chB(b);
    const int aa = chA(a); // alpha channel of `a` is preserved

    // Truncation (not rounding) matches the Kotlin reference implementation.
    const int r = clampi(ar + static_cast<int>((br - ar) * t), 0, 255);
    const int g = clampi(ag + static_cast<int>((bg - ag) * t), 0, 255);
    const int bl = clampi(ab + static_cast<int>((bb - ab) * t), 0, 255);

    return (aa << 24) | (r << 16) | (g << 8) | bl;
}

float luma(int pixel) {
    return 0.299f * chR(pixel) + 0.587f * chG(pixel) + 0.114f * chB(pixel);
}

int screen(int channel, int amount) {
    return clampi(255 - ((255 - channel) * (255 - amount)) / 255, 0, 255);
}

float smoothstep(float edge0, float edge1, float value) {
    if (edge0 == edge1) return value < edge0 ? 0.0f : 1.0f;
    const float t = clampf((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ---------------------------------------------------------------------------
// Box blur for float masks (separable, sliding window, clamp-to-edge)
// ---------------------------------------------------------------------------

void boxBlurFloat(const float* src, float* dst, int w, int h, int radius) {
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);

    if (radius <= 0) {
        if (src != dst) std::memcpy(dst, src, count * sizeof(float));
        return;
    }

    std::vector<float> tmp(count);
    const float div = static_cast<float>(2 * radius + 1);

    // Horizontal pass: src -> tmp
    for (int y = 0; y < h; ++y) {
        const float* row = src + static_cast<size_t>(y) * w;
        float* trow = tmp.data() + static_cast<size_t>(y) * w;

        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            sum += row[clampi(k, 0, w - 1)];
        }
        for (int x = 0; x < w; ++x) {
            trow[x] = sum / div;
            const int addX = clampi(x + radius + 1, 0, w - 1);
            const int subX = clampi(x - radius,     0, w - 1);
            sum += row[addX] - row[subX];
        }
    }

    // Vertical pass: tmp -> dst
    for (int x = 0; x < w; ++x) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            sum += tmp[static_cast<size_t>(clampi(k, 0, h - 1)) * w + x];
        }
        for (int y = 0; y < h; ++y) {
            dst[static_cast<size_t>(y) * w + x] = sum / div;
            const int addY = clampi(y + radius + 1, 0, h - 1);
            const int subY = clampi(y - radius,     0, h - 1);
            sum += tmp[static_cast<size_t>(addY) * w + x]
                 - tmp[static_cast<size_t>(subY) * w + x];
        }
    }
}

// ---------------------------------------------------------------------------
// Box blur for ARGB pixels (separable, sliding window, clamp-to-edge)
// ---------------------------------------------------------------------------

void gaussianBlur(const int* src, int* dst, int w, int h, int radius) {
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);

    if (radius <= 0) {
        if (src != dst) std::memcpy(dst, src, count * sizeof(int));
        return;
    }

    std::vector<int> tmp(count);
    const int div = 2 * radius + 1;

    // Horizontal pass: src -> tmp
    for (int y = 0; y < h; ++y) {
        const int* row = src + static_cast<size_t>(y) * w;
        int* trow = tmp.data() + static_cast<size_t>(y) * w;

        int rSum = 0, gSum = 0, bSum = 0;
        for (int k = -radius; k <= radius; ++k) {
            const int px = row[clampi(k, 0, w - 1)];
            rSum += chR(px);
            gSum += chG(px);
            bSum += chB(px);
        }
        for (int x = 0; x < w; ++x) {
            trow[x] = packOpaque(rSum / div, gSum / div, bSum / div);
            const int addX = clampi(x + radius + 1, 0, w - 1);
            const int subX = clampi(x - radius,     0, w - 1);
            const int add = row[addX];
            const int sub = row[subX];
            rSum += chR(add) - chR(sub);
            gSum += chG(add) - chG(sub);
            bSum += chB(add) - chB(sub);
        }
    }

    // Vertical pass: tmp -> dst
    for (int x = 0; x < w; ++x) {
        int rSum = 0, gSum = 0, bSum = 0;
        for (int k = -radius; k <= radius; ++k) {
            const int px = tmp[static_cast<size_t>(clampi(k, 0, h - 1)) * w + x];
            rSum += chR(px);
            gSum += chG(px);
            bSum += chB(px);
        }
        for (int y = 0; y < h; ++y) {
            dst[static_cast<size_t>(y) * w + x] =
                packOpaque(rSum / div, gSum / div, bSum / div);
            const int addY = clampi(y + radius + 1, 0, h - 1);
            const int subY = clampi(y - radius,     0, h - 1);
            const int add = tmp[static_cast<size_t>(addY) * w + x];
            const int sub = tmp[static_cast<size_t>(subY) * w + x];
            rSum += chR(add) - chR(sub);
            gSum += chG(add) - chG(sub);
            bSum += chB(add) - chB(sub);
        }
    }
}

// ---------------------------------------------------------------------------
// Triple box pass ≈ true Gaussian
// ---------------------------------------------------------------------------

void gaussianApprox(const int* src, int* dst, int w, int h, int radius) {
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);

    if (radius <= 0) {
        if (src != dst) std::memcpy(dst, src, count * sizeof(int));
        return;
    }

    std::vector<int> scratch(count);

    gaussianBlur(src, dst, w, h, radius);            // pass 1: src -> dst
    gaussianBlur(dst, scratch.data(), w, h, radius); // pass 2: dst -> scratch
    gaussianBlur(scratch.data(), dst, w, h, radius); // pass 3: scratch -> dst
}

} // namespace beautymath
