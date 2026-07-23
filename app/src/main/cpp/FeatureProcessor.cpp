// =============================================================================
// FeatureProcessor.cpp
// Native (NDK) facial feature enhancements — port of ApplyBeautyUseCase.kt
//
// Pixel format: ARGB_8888 packed in int32 (0xAARRGGBB), matching
// Bitmap.getPixels(). Masks are row-major float arrays [0..1], size w*h.
//
// Expected from BeautyMath.h:
//   int   blendPixel(int a, int b, float alpha);            // lerp a→b by alpha
//   float luma(int pixel);                                  // 0..255
//   int   screen(int channel, int amount);                  // screen blend, 0..255
//   void  gaussianBlur(const int* src, int* dst,
//                      int w, int h, int radius);           // one box pass
// =============================================================================

#include "BeautyMath.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

// ── Tuning constants (mirrors Kotlin companion object) ───────────────────────
constexpr float MASK_THRESHOLD         = 0.2f;
constexpr float MAX_EYE_LIFT           = 40.0f;
constexpr float MAX_UNDER_EYE_LIFT     = 60.0f;
constexpr int   SHARPEN_BLUR_RADIUS    = 2;
constexpr float MAX_SHARPEN_AMOUNT     = 0.65f;
constexpr float SHARPEN_MASK_THRESHOLD = 0.6f;

// ── Small local helpers ──────────────────────────────────────────────────────

inline int clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

inline float smoothstep(float edge0, float edge1, float value) {
    float t = (value - edge0) / (edge1 - edge0);
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return t * t * (3.0f - 2.0f * t);
}

// Separable box blur on a float mask — used to feather hard mask edges
// (equivalent of Kotlin's preBlurMask). O(w*h) via sliding window.
void featherMask(const float* src, float* dst, int w, int h, int radius) {
    if (radius <= 0) {
        std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(w) * h);
        return;
    }
    std::vector<float> tmp(static_cast<size_t>(w) * h);
    const float invWin = 1.0f / (2 * radius + 1);

    // Horizontal pass (clamped edges)
    for (int y = 0; y < h; ++y) {
        const float* row = src + static_cast<size_t>(y) * w;
        float* out = tmp.data() + static_cast<size_t>(y) * w;
        float acc = 0.0f;
        for (int x = -radius; x <= radius; ++x)
            acc += row[std::clamp(x, 0, w - 1)];
        for (int x = 0; x < w; ++x) {
            out[x] = acc * invWin;
            acc += row[std::min(x + radius + 1, w - 1)];
            acc -= row[std::max(x - radius, 0)];
        }
    }
    // Vertical pass
    for (int x = 0; x < w; ++x) {
        float acc = 0.0f;
        for (int y = -radius; y <= radius; ++y)
            acc += tmp[static_cast<size_t>(std::clamp(y, 0, h - 1)) * w + x];
        for (int y = 0; y < h; ++y) {
            dst[static_cast<size_t>(y) * w + x] = acc * invWin;
            acc += tmp[static_cast<size_t>(std::min(y + radius + 1, h - 1)) * w + x];
            acc -= tmp[static_cast<size_t>(std::max(y - radius, 0)) * w + x];
        }
    }
}

} // namespace

// =============================================================================
// 1. Under-eye bag reduction
//    Mask = max(bags - eyes, 0) * skin, feathered. Adaptive shadow lift:
//    darker pixels get the full lift, highlights get none. Red channel is
//    pulled up slightly faster to preserve skin warmth.
// =============================================================================
void applyUnderEyeReduction(int* pixels,
                            const float* bagsMask,
                            const float* eyesMask,
                            const float* skinMask,
                            int w, int h,
                            float strength) {
    const size_t n = static_cast<size_t>(w) * h;

    // 1. Combine masks: constrain bags to skin, punch out the eyeballs.
    std::vector<float> raw(n), feathered(n);
    for (size_t i = 0; i < n; ++i) {
        const float v = bagsMask[i] - eyesMask[i];
        raw[i] = (v > 0.0f ? v : 0.0f) * skinMask[i];
    }

    // 2. Gently feather the mask edge so the lift never shows a seam.
    featherMask(raw.data(), feathered.data(), w, h, 3);

    for (size_t i = 0; i < n; ++i) {
        const float maskVal = feathered[i];
        if (maskVal <= 0.01f) continue;

        const int p = pixels[i];
        const float lum = luma(p);

        // 3. Adaptive shadow lifting: inverse-luma factor, eased.
        const float shadowFactor = smoothstep(0.0f, 0.5f, 1.0f - lum / 255.0f);

        const int adaptiveLift =
            static_cast<int>(strength * MAX_UNDER_EYE_LIFT * shadowFactor);
        if (adaptiveLift <= 0) continue;

        const int a = static_cast<unsigned int>(p) >> 24;
        const int nr = ((p >> 16) & 0xFF) + adaptiveLift;
        const int ng = ((p >>  8) & 0xFF) + adaptiveLift;
        const int nb = ( p        & 0xFF) + adaptiveLift;

        // 4. Preserve skin warmth: red rises ~15% faster than G/B.
        const int warmRed =
            clamp255(static_cast<int>(nr + adaptiveLift * 0.15f));

        const int enhanced = (a << 24) | (warmRed << 16) |
                             (clamp255(ng) << 8) | clamp255(nb);

        pixels[i] = blendPixel(p, enhanced, maskVal * strength * shadowFactor);
    }
}

// =============================================================================
// 2. Eye brightness ("sparkle")
//    Gated by the feathered iris mask. luma 15..45 smoothstep detects the
//    dark pupil/iris region so only the eye itself glows — not the socket.
//    Screen blend on all three channels for the catch-light effect.
// =============================================================================
void applyEyeBrightness(int* pixels,
                        const float* irisMask,
                        int w, int h,
                        float strength) {
    const size_t n = static_cast<size_t>(w) * h;
    const int liftAmount = clamp255(static_cast<int>(strength * MAX_EYE_LIFT));

    for (size_t i = 0; i < n; ++i) {
        const float irisVal = irisMask[i];
        if (irisVal < MASK_THRESHOLD) continue;

        const int p = pixels[i];
        const float lum = luma(p);

        // Pupil/iris detection: dark pixels (luma 15–45) fade in the effect.
        const float eyeLikeness = smoothstep(15.0f, 45.0f, lum);
        if (eyeLikeness <= 0.0f) continue;

        const int a = static_cast<unsigned int>(p) >> 24;
        const int r = screen((p >> 16) & 0xFF, liftAmount);
        const int g = screen((p >>  8) & 0xFF, liftAmount);
        const int b = screen( p        & 0xFF, liftAmount);

        pixels[i] = blendPixel(p,
                               (a << 24) | (r << 16) | (g << 8) | b,
                               strength * irisVal * eyeLikeness);
    }
}

// =============================================================================
// 3. Teeth whitening
//    Redness gate: lips/gums have high R and low G; yellow teeth have high R
//    AND high G. teethLikeness = bright-luma gate * inverse-redness gate.
//    Whitening = desaturate toward the max channel (blue gets the biggest
//    boost, killing the yellow), then a screen lift for a glowing finish.
// =============================================================================
void applyTeethWhitening(int* pixels,
                         const float* teethMask,
                         int w, int h,
                         float strength) {
    const size_t n = static_cast<size_t>(w) * h;
    const int liftAmount = clamp255(static_cast<int>(strength * 100.0f));
    const float desatStrength = strength * 0.85f; // keep a hint of warmth

    for (size_t i = 0; i < n; ++i) {
        const float maskVal = teethMask[i];
        if (maskVal <= 0.0f) continue;

        const int p = pixels[i];
        const int a = static_cast<unsigned int>(p) >> 24;
        const int r = (p >> 16) & 0xFF;
        const int g = (p >>  8) & 0xFF;
        const int b =  p        & 0xFF;

        const float lum = luma(p);
        const float redness = static_cast<float>(r - g);

        // Only bright, low-redness pixels are teeth. The 50–100 window is
        // deliberately wide so stained teeth aren't misclassified as lips.
        const float teethLikeness = smoothstep(80.0f, 160.0f, lum) *
                                    (1.0f - smoothstep(50.0f, 100.0f, redness));
        if (teethLikeness <= 0.0f) continue;

        // Boost the weaker channels up toward the strongest one.
        const float maxC = static_cast<float>(std::max({r, g, b}));
        const int nr = static_cast<int>(r + (maxC - r) * desatStrength);
        const int ng = static_cast<int>(g + (maxC - g) * desatStrength);
        const int nb = static_cast<int>(b + (maxC - b) * desatStrength);

        const int lifted = (a << 24) |
                           (screen(nr, liftAmount) << 16) |
                           (screen(ng, liftAmount) <<  8) |
                            screen(nb, liftAmount);

        pixels[i] = blendPixel(p, lifted, strength * maskVal * teethLikeness);
    }
}

// =============================================================================
// 4. Face sharpening (unsharp mask)
//    out = pixel + amount * (pixel - blurred), gated by the face mask.
//    The blur source is the *original* image; writes are in-place, which is
//    safe because the blurred copy is snapshotted up front.
// =============================================================================
void applyFaceSharpening(int* pixels,
                         const float* mask,
                         int w, int h,
                         float strength) {
    const size_t n = static_cast<size_t>(w) * h;

    std::vector<int> blurred(n);
    gaussianBlur(pixels, blurred.data(), w, h, SHARPEN_BLUR_RADIUS);

    const float amount = strength * MAX_SHARPEN_AMOUNT;

    for (size_t i = 0; i < n; ++i) {
        const float maskVal = mask[i];
        if (maskVal < SHARPEN_MASK_THRESHOLD) continue;

        const int p  = pixels[i];
        const int bp = blurred[i];

        const int a  = static_cast<unsigned int>(p) >> 24;
        const int pr = (p >> 16) & 0xFF, br = (bp >> 16) & 0xFF;
        const int pg = (p >>  8) & 0xFF, bg = (bp >>  8) & 0xFF;
        const int pb =  p        & 0xFF, bb =  bp        & 0xFF;

        const int r = clamp255(static_cast<int>(pr + amount * (pr - br)));
        const int g = clamp255(static_cast<int>(pg + amount * (pg - bg)));
        const int b = clamp255(static_cast<int>(pb + amount * (pb - bb)));

        pixels[i] = blendPixel(p,
                               (a << 24) | (r << 16) | (g << 8) | b,
                               maskVal * strength);
    }
}
