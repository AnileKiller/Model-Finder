#ifndef BEAUTY_MATH_H
#define BEAUTY_MATH_H

#include <cstdint>

/**
 * BeautyMath — low-level image math for the beauty camera engine.
 *
 * Pure C++ / NDK. No OpenCV, no external libraries.
 *
 * Conventions:
 *  - Pixels are ARGB_8888 packed into a 32-bit int: (A << 24) | (R << 16) | (G << 8) | B.
 *    This matches android.graphics.Bitmap.Config.ARGB_8888 as read via Bitmap.getPixels().
 *  - Images are 1D row-major arrays of length w * h (index = y * w + x).
 *  - Masks are float arrays in the range 0.0 .. 1.0.
 *  - Blur functions use clamp-to-edge sampling and a sliding-window accumulator,
 *    so cost is O(w * h) and independent of radius.
 *  - All functions are thread-safe (no shared state); src and dst must not alias
 *    unless documented otherwise.
 */
namespace beautymath {

/**
 * Linear blend of two ARGB pixels: result = a + (b - a) * alpha per RGB channel.
 * The alpha (A) channel of pixel `a` is preserved. `alpha` is clamped to [0, 1].
 */
int blendPixel(int a, int b, float alpha);

/** Rec.601 luminance of an ARGB pixel: 0.299*R + 0.587*G + 0.114*B. Range 0.0 .. 255.0. */
float luma(int pixel);

/** Screen blend for a single 0-255 channel: 255 - (255 - channel) * (255 - amount) / 255. */
int screen(int channel, int amount);

/** Standard smoothstep: Hermite interpolation of `value` between edge0 and edge1, clamped. */
float smoothstep(float edge0, float edge1, float value);

/**
 * Fast separable box blur (horizontal pass, then vertical pass) for float mask arrays.
 * Clamp-to-edge boundary handling. O(w * h), independent of radius.
 * If radius <= 0, src is copied to dst unchanged.
 * src and dst must be distinct buffers of length w * h.
 */
void boxBlurFloat(const float* src, float* dst, int w, int h, int radius);

/**
 * Fast separable box blur (single pass: horizontal then vertical) for ARGB pixel arrays,
 * used as a cheap Gaussian approximation. Output alpha is forced to 0xFF (opaque).
 * Clamp-to-edge boundary handling. O(w * h), independent of radius.
 * If radius <= 0, src is copied to dst unchanged.
 * src and dst must be distinct buffers of length w * h.
 */
void gaussianBlur(const int* src, int* dst, int w, int h, int radius);

/**
 * Runs gaussianBlur three times back-to-back. Per the central limit theorem,
 * three box passes closely approximate a true Gaussian curve.
 * src and dst must be distinct buffers of length w * h.
 */
void gaussianApprox(const int* src, int* dst, int w, int h, int radius);

} // namespace beautymath

#endif // BEAUTY_MATH_H
