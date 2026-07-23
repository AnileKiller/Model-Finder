// =============================================================================
// BlemishEngine.cpp
//
// Frequency-separation blemish removal — C++ (NDK) port of the Kotlin
// ApplyBeautyUseCase::applyBlemishReduction pipeline.
//
// Blemishes are treated as two separate problems:
//
//   1. Low-frequency colour/tone defects (redness, purple/brown inflammatory
//      marks, broad acne-cluster patches) — repaired by pulling the blurred
//      skin base toward a clean-weighted donor sampled ONLY from non-suspect
//      skin, so lesions can never donate their own colour back.
//
//   2. High-frequency texture defects (raised pimple centres, sharp scabs) —
//      reduced by attenuating the detail layer while keeping normal skin
//      texture, so the result never turns waxy.
//
// Assumed available from BeautyMath.h (all buffers are 1D, row-major, ARGB):
//
//   void  gaussianApprox(const int* src, int* dst, int w, int h, int radius);
//   void  boxBlurFloat  (const float* src, float* dst, int w, int h, int radius);
//   float smoothstep    (float edge0, float edge1, float x);
//   int   blendPixel    (int base, int overlay, float alpha);   // preserves base alpha
//   float luma          (int argb);                              // BT.601 luma
// =============================================================================

#include "BeautyMath.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifndef LOG_TAG
#define LOG_TAG "BlemishEngine"
#endif
#ifdef __ANDROID__
#include <android/log.h>
#define BLOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
#define BLOGD(...) ((void)0)
#endif

namespace beauty {

// ---------------------------------------------------------------------------
// Tunables (mirrors Kotlin companion-object constants)
// ---------------------------------------------------------------------------
static constexpr float BLEMISH_MAX_ALPHA = 0.985f;

// ---------------------------------------------------------------------------
// Small local helpers
// ---------------------------------------------------------------------------
static inline int   clampi(int v, int lo, int hi)     { return v < lo ? lo : (v > hi ? hi : v); }
static inline float clampf(float v, float lo, float hi){ return v < lo ? lo : (v > hi ? hi : v); }

static inline int R(int p) { return (p >> 16) & 0xFF; }
static inline int G(int p) { return (p >>  8) & 0xFF; }
static inline int B(int p) { return  p        & 0xFF; }

static inline int packRGB(int alphaSrc, int r, int g, int b) {
    return (alphaSrc & 0xFF000000) | (r << 16) | (g << 8) | b;
}

// =============================================================================
// cleanWeightedBlurTarget
//
// Builds a skin-colour donor image by blurring only clean-weighted pixels
// (a normalized-convolution / "cheap inpainting" pass). Where the blurred
// weight field has too little support, falls back to the supplied target.
// =============================================================================
void cleanWeightedBlurTarget(const int* src,
                             const float* cleanWeight,
                             const int* fallback,
                             int* dst,
                             int w, int h, int radius) {
    const int n = w * h;

    if (radius <= 0) {
        // Degenerate radius: pass source through where weighted, else fallback.
        for (int i = 0; i < n; ++i) {
            dst[i] = (cleanWeight[i] > 0.035f) ? src[i] : fallback[i];
        }
        return;
    }

    // Pre-multiply each channel by its clean weight.
    std::vector<float> wr(n), wg(n), wb(n);
    for (int i = 0; i < n; ++i) {
        const float weight = clampf(cleanWeight[i], 0.0f, 1.0f);
        const int p = src[i];
        wr[i] = static_cast<float>(R(p)) * weight;
        wg[i] = static_cast<float>(G(p)) * weight;
        wb[i] = static_cast<float>(B(p)) * weight;
    }

    // Blur weights and weighted channels with the same kernel, then normalize.
    std::vector<float> bw(n), br(n), bg(n), bb(n);
    boxBlurFloat(cleanWeight, bw.data(), w, h, radius);
    boxBlurFloat(wr.data(),   br.data(), w, h, radius);
    boxBlurFloat(wg.data(),   bg.data(), w, h, radius);
    boxBlurFloat(wb.data(),   bb.data(), w, h, radius);

    for (int i = 0; i < n; ++i) {
        const float weight = bw[i];
        if (weight > 0.035f) {
            const float inv = 1.0f / weight;
            const int r = clampi(static_cast<int>(br[i] * inv), 0, 255);
            const int g = clampi(static_cast<int>(bg[i] * inv), 0, 255);
            const int b = clampi(static_cast<int>(bb[i] * inv), 0, 255);
            dst[i] = packRGB(src[i], r, g, b);
        } else {
            // Too little clean support nearby — trust the wider fallback donor.
            dst[i] = fallback[i];
        }
    }
}

// =============================================================================
// maxFilterMask
//
// 2D max filter (grayscale dilation) with clamped edges over a 1D float mask.
// A square structuring element is separable, so we run two 1D passes
// (horizontal then vertical): O(n·r) instead of O(n·r²).
// =============================================================================
void maxFilterMask(const float* mask, float* dst, int w, int h, int radius) {
    const int n = w * h;
    if (radius <= 0) {
        std::memcpy(dst, mask, static_cast<size_t>(n) * sizeof(float));
        return;
    }

    std::vector<float> tmp(n);

    // Horizontal pass.
    for (int y = 0; y < h; ++y) {
        const float* row = mask + y * w;
        float* out = tmp.data() + y * w;
        for (int x = 0; x < w; ++x) {
            float m = 0.0f;
            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(w - 1, x + radius);
            for (int xx = x0; xx <= x1; ++xx) {
                if (row[xx] > m) m = row[xx];
            }
            out[x] = m;
        }
    }

    // Vertical pass.
    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(h - 1, y + radius);
        float* out = dst + y * w;
        for (int x = 0; x < w; ++x) {
            float m = 0.0f;
            for (int yy = y0; yy <= y1; ++yy) {
                const float v = tmp[yy * w + x];
                if (v > m) m = v;
            }
            out[x] = m;
        }
    }
}

// =============================================================================
// applyBlemishReduction
//
// Full adaptive pipeline. `pixels` is modified in place (ARGB_8888 as int),
// `mask` is the 0..1 skin/face mask (ocular shield already applied upstream).
// =============================================================================
void applyBlemishReduction(int* pixels,
                           const float* mask,
                           int w, int h,
                           float strength) {
    const int n = w * h;
    if (n <= 0 || strength <= 0.0f) return;

    // Keep an untouched copy: detection and reconstruction both read the
    // original while we write the result back into `pixels`.
    std::vector<int> original(pixels, pixels + n);

    // -----------------------------------------------------------------------
    // Step 1 — Median facial redness (R-G) via histogram. This adapts the
    // detector to complexion and white balance instead of a fixed threshold.
    // -----------------------------------------------------------------------
    int facePixels = 0;
    int redHistogram[129] = {0};                 // R-G in [-64, 64]
    for (int i = 0; i < n; ++i) {
        if (mask[i] > 0.50f) {
            ++facePixels;
            const int p = original[i];
            const int rg = clampi(R(p) - G(p), -64, 64);
            ++redHistogram[rg + 64];
        }
    }
    if (facePixels == 0) return;                 // No face — nothing to repair.

    const float faceScale = std::max(std::sqrt(static_cast<float>(facePixels)), 80.0f);

    int medianRg = 0;
    {
        int cumulative = 0;
        const int half = (facePixels + 1) / 2;
        for (int bin = 0; bin < 129; ++bin) {
            cumulative += redHistogram[bin];
            if (cumulative >= half) { medianRg = bin - 64; break; }
        }
    }

    // -----------------------------------------------------------------------
    // Step 2 — Three frequency scales: texture, individual lesions, and
    // diffuse acne clusters. All radii scale with face size.
    // -----------------------------------------------------------------------
    const int detailRadius = clampi(static_cast<int>(faceScale * 0.0045f),  1,  3);
    const int localRadius  = clampi(static_cast<int>(faceScale * 0.018f),   5, 12);
    const int broadRadius  = clampi(static_cast<int>(faceScale * 0.055f),  16, 36);
    const int donorRadius  = clampi(static_cast<int>(faceScale * 0.065f),  18, 42);
    const int growRadius   = clampi(static_cast<int>(faceScale * 0.006f),   2,  4);

    std::vector<int> detailBase(n), localBase(n), broadBase(n);
    gaussianApprox(original.data(), detailBase.data(), w, h, detailRadius);
    gaussianApprox(original.data(), localBase.data(),  w, h, localRadius);
    gaussianApprox(original.data(), broadBase.data(),  w, h, broadRadius);

    // -----------------------------------------------------------------------
    // Step 3 — Per-pixel confidence map. Detection is deliberately independent
    // of UI strength: otherwise low strength lets blemishes back into the
    // donor pool and contaminates their own replacement colour.
    // -----------------------------------------------------------------------
    std::vector<float> confidence(n, 0.0f);
    int coreHits = 0;

    for (int i = 0; i < n; ++i) {
        const float maskValue = mask[i];
        if (maskValue < 0.08f) continue;

        const int p  = original[i];
        const int lp = localBase[i];
        const int bp = broadBase[i];
        const int dp = detailBase[i];

        const int r  = R(p),  g  = G(p),  b  = B(p);
        const int lr = R(lp), lg = G(lp), lb = B(lp);
        const int br = R(bp), bg = G(bp), bb = B(bp);

        const float y0 = 0.299f * r  + 0.587f * g  + 0.114f * b;
        const float yl = 0.299f * lr + 0.587f * lg + 0.114f * lb;
        const float yb = 0.299f * br + 0.587f * bg + 0.114f * bb;
        const float yd = luma(dp);

        // Red residual at two scales catches isolated pimples AND dense clusters.
        const float rg = static_cast<float>(r - g);
        const float localRed      = rg - static_cast<float>(lr - lg);
        const float broadRed      = rg - static_cast<float>(br - bg);
        const float complexionRed = rg - static_cast<float>(medianRg);

        const float focalRedScore = std::max(
            smoothstep(2.5f, 18.0f, localRed),
            smoothstep(4.0f, 24.0f, broadRed));
        const float complexionRedScore = smoothstep(12.0f, 38.0f, complexionRed);

        // Purple/brown inflammatory marks often carry little R-G residual —
        // measure chroma displacement from the broad skin field instead.
        const float purpleNow   = (r + b) * 0.5f - g;
        const float purpleBroad = (br + bb) * 0.5f - bg;
        const float chromaShift =
            std::fabs(static_cast<float>((r - g) - (br - bg))) +
            0.55f * std::fabs(static_cast<float>((b - g) - (bb - bg)));
        const float purpleScore = smoothstep(5.0f, 24.0f, purpleNow - purpleBroad);
        const float chromaScore = smoothstep(9.0f, 32.0f, chromaShift);

        const float darkResidual = std::max(yl - y0, yb - y0);
        const float darkScore    = smoothstep(6.0f, 25.0f, darkResidual);
        const float textureScore = smoothstep(2.5f, 14.0f, std::fabs(y0 - yd));

        // Complexion-relative redness is only supporting evidence, never a
        // standalone score — protects naturally rosy cheeks / warm makeup.
        const float redScore = std::max(
            focalRedScore,
            complexionRedScore * std::max({focalRedScore,
                                           textureScore * 0.65f,
                                           darkScore * 0.55f}));

        // Darkness/compact contrast must also show inflammatory chroma; this
        // intentionally preserves neutral freckles and moles.
        const float inflammatoryEvidence =
            std::max({redScore, purpleScore, chromaScore * 0.70f});
        const float darkInflamed  = darkScore * inflammatoryEvidence;
        const float compactDefect = darkScore * textureScore * inflammatoryEvidence * 0.62f;

        const float score = std::max({redScore, purpleScore * 0.82f,
                                      darkInflamed, compactDefect});

        confidence[i] = clampf(score * maskValue, 0.0f, 1.0f);
        if (score > 0.12f) ++coreHits;
    }

    // -----------------------------------------------------------------------
    // Step 4 — Grow the detected core to cover the whole lesion, then feather
    // the boundary so the repair blends invisibly.
    // -----------------------------------------------------------------------
    std::vector<float> grown(n), repairMask(n);
    maxFilterMask(confidence.data(), grown.data(), w, h, growRadius);
    boxBlurFloat(grown.data(), repairMask.data(), w, h, growRadius);

    // -----------------------------------------------------------------------
    // Step 5 — Clean donor image. Weights come only from pixels that are both
    // skin and confidently clean; the squared rejection term makes suspect
    // pixels drop out of the donor pool fast. A double-radius clean pass is
    // the fallback for dense clusters; only its final emergency fallback
    // contains unfiltered broad colour.
    // -----------------------------------------------------------------------
    std::vector<float> cleanWeight(n);
    for (int i = 0; i < n; ++i) {
        const float skin = clampf(mask[i], 0.0f, 1.0f);
        const float rejected = clampf(std::max(grown[i], confidence[i]), 0.0f, 1.0f);
        cleanWeight[i] = (skin > 0.08f)
            ? skin * (1.0f - rejected) * (1.0f - rejected)
            : 0.0f;
    }

    std::vector<int> farDonor(n), donor(n);
    cleanWeightedBlurTarget(detailBase.data(), cleanWeight.data(),
                            broadBase.data(), farDonor.data(),
                            w, h, std::min(donorRadius * 2, 72));
    cleanWeightedBlurTarget(detailBase.data(), cleanWeight.data(),
                            farDonor.data(), donor.data(),
                            w, h, donorRadius);

    // -----------------------------------------------------------------------
    // Step 6 — Reconstruct: donor low-frequency base + controlled original
    // high-frequency texture. Weak detections keep more texture; severe
    // lesion cores keep less.
    // -----------------------------------------------------------------------
    const float effectStrength = clampf(strength, 0.0f, 1.0f);
    int repairedPixels = 0;
    float alphaSum = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float confidenceValue = clampf(repairMask[i], 0.0f, 1.0f);
        const float alpha = clampf(confidenceValue * mask[i] * effectStrength,
                                   0.0f, BLEMISH_MAX_ALPHA);
        if (alpha <= 0.015f) continue;

        const int o      = original[i];
        const int base   = detailBase[i];
        const int target = donor[i];

        const int oR = R(o),      oG = G(o),      oB = B(o);
        const int bR = R(base),   bG = G(base),   bB = B(base);
        const int tR = R(target), tG = G(target), tB = B(target);

        const float textureKeep =
            clampf(0.52f - 0.30f * confidenceValue, 0.20f, 0.52f);

        const int rr = clampi(static_cast<int>(tR + (oR - bR) * textureKeep), 0, 255);
        const int rg = clampi(static_cast<int>(tG + (oG - bG) * textureKeep), 0, 255);
        const int rb = clampi(static_cast<int>(tB + (oB - bB) * textureKeep), 0, 255);

        const int repaired = packRGB(o, rr, rg, rb);
        pixels[i] = blendPixel(o, repaired, alpha);
        ++repairedPixels;
        alphaSum += alpha;
    }

    BLOGD("Blemish adaptive: cores=%d repaired=%d meanAlpha=%.3f medianRG=%d "
          "radii=%d/%d/%d/%d grow=%d",
          coreHits, repairedPixels,
          repairedPixels > 0 ? alphaSum / repairedPixels : 0.0f,
          medianRg, detailRadius, localRadius, broadRadius, donorRadius,
          growRadius);
}

} // namespace beauty
