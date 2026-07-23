// =============================================================================
// SkinProcessor.cpp
//
// Base skin-beautification modules for the Android NDK beauty-camera pipeline.
// Ported from ApplyBeautyUseCase.kt (Kotlin) to optimized C++.
//
// Conventions:
//   * pixels : 1D int array, ARGB_8888 packed (A << 24 | R << 16 | G << 8 | B)
//   * mask   : 1D float array [0..1], same dimensions, skin probability
//   * All functions operate in-place on `pixels`.
//
// Core math helpers (blendPixel, gaussianBlur, etc.) live in BeautyMath.h.
// =============================================================================

#include "BeautyMath.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

namespace beauty {

// -----------------------------------------------------------------------------
// Local helpers (kept in an anonymous namespace to avoid symbol pollution)
// -----------------------------------------------------------------------------
namespace {

// Mask threshold below which a pixel is skipped entirely. Matches the Kotlin
// pipeline's "skip non-skin instantly" fast path.
constexpr float kMaskSkipThreshold = 0.2f;

// Fraction of the high-frequency (texture) signal blended back after the
// bilateral pass so pores/detail survive the smoothing.
constexpr float kTextureRetain = 0.28f;

// Bilateral filter tuning.
constexpr int   kBilateralRadius     = 4;      // 9x9 kernel
constexpr float kBilateralSigmaSpace = 3.0f;
constexpr float kBilateralSigmaColor = 30.0f;  // range sigma (0..255 domain)

// Matches MAX_BRIGHTNESS_LIFT in the Kotlin implementation.
constexpr float kMaxBrightnessLift = 80.0f;

inline int clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

inline int packARGB(int a, int r, int g, int b) {
    return (a << 24) | (r << 16) | (g << 8) | b;
}

// Fast luminance approximation: (R*2 + G*5 + B) / 8. Integer-only, good enough
// as the bilateral range key (avoids per-channel range weights: 3x cheaper).
inline int fastLuma(int c) {
    const int r = (c >> 16) & 0xFF;
    const int g = (c >> 8)  & 0xFF;
    const int b =  c        & 0xFF;
    return (r * 2 + g * 5 + b) >> 3;
}

// RGB [0..255] -> HSV (h in [0..360), s/v in [0..1]).
inline void rgbToHsv(int r, int g, int b, float& h, float& s, float& v) {
    const float rf = r * (1.0f / 255.0f);
    const float gf = g * (1.0f / 255.0f);
    const float bf = b * (1.0f / 255.0f);

    const float maxC = std::max(rf, std::max(gf, bf));
    const float minC = std::min(rf, std::min(gf, bf));
    const float delta = maxC - minC;

    v = maxC;
    s = (maxC > 0.0f) ? (delta / maxC) : 0.0f;

    if (delta < 1e-6f) {
        h = 0.0f;
    } else if (maxC == rf) {
        h = 60.0f * std::fmod((gf - bf) / delta, 6.0f);
    } else if (maxC == gf) {
        h = 60.0f * ((bf - rf) / delta + 2.0f);
    } else {
        h = 60.0f * ((rf - gf) / delta + 4.0f);
    }
    if (h < 0.0f) h += 360.0f;
}

// HSV -> RGB [0..255].
inline void hsvToRgb(float h, float s, float v, int& r, int& g, int& b) {
    const float c = v * s;
    const float hp = h * (1.0f / 60.0f);
    const float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
    const float m = v - c;

    float rf, gf, bf;
    if      (hp < 1.0f) { rf = c;  gf = x;  bf = 0;  }
    else if (hp < 2.0f) { rf = x;  gf = c;  bf = 0;  }
    else if (hp < 3.0f) { rf = 0;  gf = c;  bf = x;  }
    else if (hp < 4.0f) { rf = 0;  gf = x;  bf = c;  }
    else if (hp < 5.0f) { rf = x;  gf = 0;  bf = c;  }
    else                { rf = c;  gf = 0;  bf = x;  }

    r = clamp255(static_cast<int>((rf + m) * 255.0f + 0.5f));
    g = clamp255(static_cast<int>((gf + m) * 255.0f + 0.5f));
    b = clamp255(static_cast<int>((bf + m) * 255.0f + 0.5f));
}

} // anonymous namespace

// =============================================================================
// 1. Skin Smoothing — fast bilateral filter with texture re-injection
// =============================================================================
//
// Strategy:
//   * Pre-compute the (2r+1)^2 spatial Gaussian weights once per call.
//   * Pre-compute a 256-entry LUT for the range (color) Gaussian so the inner
//     loop does two table lookups + one multiply, no expf().
//   * Use luminance as the range key (single LUT lookup instead of three).
//   * Read from an untouched copy of the source so the filter is not
//     order-dependent (in-place bilateral causes directional streaking).
//   * High-frequency detail = source - smoothed; add 28% of it back so pores
//     and fine texture are preserved ("frequency separation" look).
//
void applySkinSmoothing(int* pixels, const float* mask, int w, int h, float strength) {
    if (strength <= 0.0f || w <= 0 || h <= 0) return;
    strength = std::min(strength, 1.0f);

    constexpr int r    = kBilateralRadius;
    constexpr int side = 2 * r + 1;

    // ---- Pre-compute spatial weight matrix (side x side) --------------------
    float spatial[side * side];
    {
        const float invTwoSigmaSp2 =
            1.0f / (2.0f * kBilateralSigmaSpace * kBilateralSigmaSpace);
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                spatial[(dy + r) * side + (dx + r)] =
                    std::exp(-static_cast<float>(dx * dx + dy * dy) * invTwoSigmaSp2);
            }
        }
    }

    // ---- Pre-compute range weight LUT (|deltaLuma| in 0..255) ---------------
    float rangeLUT[256];
    {
        const float invTwoSigmaCol2 =
            1.0f / (2.0f * kBilateralSigmaColor * kBilateralSigmaColor);
        for (int d = 0; d < 256; ++d) {
            rangeLUT[d] = std::exp(-static_cast<float>(d * d) * invTwoSigmaCol2);
        }
    }

    // ---- Immutable source copy ----------------------------------------------
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<int> src(pixels, pixels + count);
    const int* s = src.data();

    // ---- Filter loop ---------------------------------------------------------
    for (int y = 0; y < h; ++y) {
        const int rowBase = y * w;
        for (int x = 0; x < w; ++x) {
            const int idx = rowBase + x;

            const float m = mask[idx];
            if (m < kMaskSkipThreshold) continue;   // non-skin: untouched

            const int center     = s[idx];
            const int centerLuma = fastLuma(center);
            const int a          = (center >> 24) & 0xFF;

            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumW = 0.0f;

            const int y0 = std::max(y - r, 0),  y1 = std::min(y + r, h - 1);
            const int x0 = std::max(x - r, 0),  x1 = std::min(x + r, w - 1);

            for (int ny = y0; ny <= y1; ++ny) {
                const int* srow   = s + ny * w;
                const float* wrow = spatial + (ny - y + r) * side + (r - x);
                for (int nx = x0; nx <= x1; ++nx) {
                    const int c = srow[nx];
                    const int dLuma = fastLuma(c) - centerLuma;
                    const float wgt =
                        wrow[nx] * rangeLUT[dLuma < 0 ? -dLuma : dLuma];

                    sumR += wgt * ((c >> 16) & 0xFF);
                    sumG += wgt * ((c >> 8)  & 0xFF);
                    sumB += wgt * ( c        & 0xFF);
                    sumW += wgt;
                }
            }

            const float invW = 1.0f / sumW;
            const float smR = sumR * invW;
            const float smG = sumG * invW;
            const float smB = sumB * invW;

            const int srcR = (center >> 16) & 0xFF;
            const int srcG = (center >> 8)  & 0xFF;
            const int srcB =  center        & 0xFF;

            // Re-inject 28% of the high-frequency texture (src - smoothed).
            const float outRf = smR + kTextureRetain * (srcR - smR);
            const float outGf = smG + kTextureRetain * (srcG - smG);
            const float outBf = smB + kTextureRetain * (srcB - smB);

            // Overall blend: strength x mask feathering.
            const float t = strength * m;
            const int outR = clamp255(static_cast<int>(srcR + t * (outRf - srcR) + 0.5f));
            const int outG = clamp255(static_cast<int>(srcG + t * (outGf - srcG) + 0.5f));
            const int outB = clamp255(static_cast<int>(srcB + t * (outBf - srcB) + 0.5f));

            pixels[idx] = packARGB(a, outR, outG, outB);
        }
    }
}

// =============================================================================
// 2. Skin Brightness — HSV Value-channel screen lift
// =============================================================================
//
// Mirrors the Kotlin applySkinBrightness: lift V with a "screen" curve
//   V' = V + lift * (1 - V)
// so bright regions saturate gently instead of clipping, and shadow depth
// survives. Blend result against the source with the mask.
//
void applySkinBrightness(int* pixels, const float* mask, int w, int h, float strength) {
    if (strength <= 0.0f || w <= 0 || h <= 0) return;
    strength = std::min(strength, 1.0f);

    const float lift = strength * (kMaxBrightnessLift / 255.0f); // 0..1 V-lift

    const int count = w * h;
    for (int idx = 0; idx < count; ++idx) {
        const float m = mask[idx];
        if (m <= 0.0f) continue;

        const int c = pixels[idx];
        const int a =  (c >> 24) & 0xFF;
        const int r0 = (c >> 16) & 0xFF;
        const int g0 = (c >> 8)  & 0xFF;
        const int b0 =  c        & 0xFF;

        // RGB -> HSV, lift V, HSV -> RGB.
        float hh, ss, vv;
        rgbToHsv(r0, g0, b0, hh, ss, vv);
        vv = std::min(1.0f, std::max(0.0f, vv + lift * (1.0f - vv)));

        int r1, g1, b1;
        hsvToRgb(hh, ss, vv, r1, g1, b1);

        // Mask-weighted blend back onto the original pixel.
        const int outR = clamp255(static_cast<int>(r0 + m * (r1 - r0) + 0.5f));
        const int outG = clamp255(static_cast<int>(g0 + m * (g1 - g0) + 0.5f));
        const int outB = clamp255(static_cast<int>(b0 + m * (b1 - b0) + 0.5f));

        pixels[idx] = packARGB(a, outR, outG, outB);
    }
}

// =============================================================================
// 3. Skin Tone — warm-shift pixel offset
// =============================================================================
//
// R += strength * 6, B -= strength * 5. The fixed offsets are folded into the
// mask multiply so the whole inner loop is two adds, two clamps, one pack.
//
void applySkinTone(int* pixels, const float* mask, int w, int h, float strength) {
    if (strength <= 0.0f || w <= 0 || h <= 0) return;
    strength = std::min(strength, 1.0f);

    const float rOffset = strength * 6.0f;   // warm the reds
    const float bOffset = strength * 5.0f;   // pull down the blues

    const int count = w * h;
    for (int idx = 0; idx < count; ++idx) {
        const float m = mask[idx];
        if (m <= 0.0f) continue;

        const int c = pixels[idx];
        const int a = (c >> 24) & 0xFF;
        const int r = (c >> 16) & 0xFF;
        const int g = (c >> 8)  & 0xFF;
        const int b =  c        & 0xFF;

        // Offsets scaled by the mask == blending the shifted pixel by mask.
        const int outR = clamp255(static_cast<int>(r + m * rOffset + 0.5f));
        const int outB = clamp255(static_cast<int>(b - m * bOffset + 0.5f));

        pixels[idx] = packARGB(a, outR, g, outB);
    }
}

} // namespace beauty
