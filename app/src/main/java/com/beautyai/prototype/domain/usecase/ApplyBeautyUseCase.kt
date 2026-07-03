package com.beautyai.prototype.domain.usecase

import android.graphics.Bitmap
import android.graphics.RectF
import com.beautyai.prototype.domain.model.BeautyParameters
import com.beautyai.prototype.domain.model.FaceData
import kotlin.math.sqrt

/**
 * Applies the full beauty-enhancement pipeline to [source] using the pre-computed
 * [faceData] and user-controlled [params].
 *
 * Processing runs on whatever coroutine dispatcher the caller provides
 * (use [Dispatchers.Default] for background work).
 *
 * ## Key fixes vs. v1
 * 1. Skin smoothing now uses a bilateral filter (edge-preserving) instead of
 *    Gaussian blur, so eyes/lips/nose outline survive even at max strength.
 *    After smoothing, ~25 % of the original high-frequency detail is blended
 *    back in so skin retains micro-texture rather than looking plastic.
 * 2. Skin brightness works in HSL space and only raises Lightness, preserving
 *    warm hue and saturation.  Max lift is capped so highlights don't blow out.
 * 3. Eye brightness is now luminance-gated (same trick as teeth whitening) so
 *    it only brightens the whites/iris, not eyelids or surrounding skin.
 * 4. Face sharpening amount is capped and only applied to high-confidence skin
 *    pixels to avoid halos at mask boundaries.
 * 5. Pipeline order: sharpen first (on the unmodified original detail), then
 *    smooth — so sharpening doesn't amplify artefacts from other effects.
 */
class ApplyBeautyUseCase {

    operator fun invoke(
        source: Bitmap,
        faceData: FaceData,
        params: BeautyParameters
    ): Bitmap {
        val effective = params.withGlobalIntensity()
        var result = source.copy(Bitmap.Config.ARGB_8888, true)

        // Sharpening first — works on original, unmodified detail
        if (effective.faceSharpening > 0f)
            result = applyFaceSharpening(result, faceData, effective.faceSharpening)

        if (effective.skinSmoothing > 0f)
            result = applySkinSmoothing(result, faceData, effective.skinSmoothing)

        if (effective.blemishReduction > 0f)
            result = applyBlemishReduction(result, faceData, effective.blemishReduction)

        if (effective.skinBrightness > 0f)
            result = applySkinBrightness(result, faceData, effective.skinBrightness)

        if (effective.skinToneEnhancement > 0f)
            result = applySkinTone(result, faceData, effective.skinToneEnhancement)

        if (effective.underEyeReduction > 0f)
            result = applyUnderEyeReduction(result, faceData, effective.underEyeReduction)

        if (effective.eyeBrightness > 0f)
            result = applyEyeBrightness(result, faceData, effective.eyeBrightness)

        if (effective.teethWhitening > 0f)
            result = applyTeethWhitening(result, faceData, effective.teethWhitening)

        return result
    }

    // ── Individual beauty effects ────────────────────────────────────────────

    /**
     * Bilateral-filter skin smoothing — edge-preserving.
     *
     * For each skin pixel the filter averages neighbours weighted by BOTH
     * spatial proximity (Gaussian kernel) AND colour similarity (range
     * Gaussian).  Pixels that differ significantly in colour from the centre
     * contribute very little, so sharp edges (eye outline, lip border, nose)
     * survive even at max [strength].
     *
     * After the bilateral pass, ~[TEXTURE_PRESERVE_AMOUNT] of the original
     * high-frequency detail (original − blurred) is composited back in, so
     * skin retains pore-level micro-texture rather than looking like plastic.
     *
     * The spatial radius scales with [strength]: 1 px at 0.08 → 6 px at 1.0.
     * The colour sigma is fixed; lowering it makes the filter more edge-aware.
     */
    private fun applySkinSmoothing(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val radius = (strength * MAX_BILATERAL_RADIUS).toInt().coerceAtLeast(1)
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        // Build precomputed spatial Gaussian weights for the kernel
        val spatialWeights = FloatArray((radius + 1) * (radius + 1))
        for (dy in 0..radius) for (dx in 0..radius) {
            val dist2 = (dx * dx + dy * dy).toFloat()
            spatialWeights[dy * (radius + 1) + dx] =
                kotlin.math.exp(-dist2 / (2f * BILATERAL_SPATIAL_SIGMA * BILATERAL_SPATIAL_SIGMA))
        }

        val bilateral = IntArray(w * h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = face.segmentationMask[y][x]
                if (maskVal < MASK_THRESHOLD) {
                    bilateral[y * w + x] = pixels[y * w + x]
                    continue
                }
                val cp = pixels[y * w + x]
                val cr = cp shr 16 and 0xFF
                val cg = cp shr 8  and 0xFF
                val cb = cp        and 0xFF

                var rAcc = 0f; var gAcc = 0f; var bAcc = 0f; var wAcc = 0f

                for (dy in -radius..radius) {
                    val ny = (y + dy).coerceIn(0, h - 1)
                    for (dx in -radius..radius) {
                        val nx = (x + dx).coerceIn(0, w - 1)
                        val np = pixels[ny * w + nx]
                        val nr = np shr 16 and 0xFF
                        val ng = np shr 8  and 0xFF
                        val nb = np        and 0xFF

                        val colorDiff = (cr - nr) * (cr - nr) +
                                        (cg - ng) * (cg - ng) +
                                        (cb - nb) * (cb - nb)
                        val rangW = kotlin.math.exp(
                            -colorDiff / (2f * BILATERAL_COLOR_SIGMA * BILATERAL_COLOR_SIGMA)
                        )
                        val sw = spatialWeights[
                            kotlin.math.abs(dy) * (radius + 1) + kotlin.math.abs(dx)
                        ]
                        val w2 = sw * rangW

                        rAcc += nr * w2
                        gAcc += ng * w2
                        bAcc += nb * w2
                        wAcc += w2
                    }
                }

                if (wAcc > 0f) {
                    val a = cp ushr 24
                    val smoothed = (a shl 24) or
                            ((rAcc / wAcc).toInt().coerceIn(0, 255) shl 16) or
                            ((gAcc / wAcc).toInt().coerceIn(0, 255) shl 8) or
                            (bAcc / wAcc).toInt().coerceIn(0, 255)
                    bilateral[y * w + x] = blendPixel(cp, smoothed, maskVal * strength)
                } else {
                    bilateral[y * w + x] = cp
                }
            }
        }

        // Blend back high-frequency detail to preserve skin texture
        val out = IntArray(w * h)
        for (i in pixels.indices) {
            val maskY = i / w; val maskX = i % w
            val maskVal = face.segmentationMask.getOrNull(maskY)?.getOrNull(maskX) ?: 0f
            if (maskVal < MASK_THRESHOLD) { out[i] = bilateral[i]; continue }

            // HF detail = original − bilateral result
            val orig = pixels[i]; val bil = bilateral[i]
            val a = orig ushr 24
            val detailR = ((orig shr 16 and 0xFF) - (bil shr 16 and 0xFF))
            val detailG = ((orig shr 8  and 0xFF) - (bil shr 8  and 0xFF))
            val detailB = ((orig        and 0xFF) - (bil        and 0xFF))

            val textureStr = TEXTURE_PRESERVE_AMOUNT * strength  // less texture at low strength
            val r = ((bil shr 16 and 0xFF) + detailR * textureStr).toInt().coerceIn(0, 255)
            val g = ((bil shr 8  and 0xFF) + detailG * textureStr).toInt().coerceIn(0, 255)
            val b = ((bil        and 0xFF) + detailB * textureStr).toInt().coerceIn(0, 255)
            out[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }

        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Skin brightness in HSL space — only Lightness is raised.
     *
     * Working in HSL means the warm hue and saturation of skin are untouched;
     * the image looks brighter without going pale/grey.  Max lift is capped at
     * [MAX_LIGHTNESS_LIFT] (0–1) so highlights don't blow out even at
     * strength = 1.0.
     */
    private fun applySkinBrightness(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val liftFraction = strength * MAX_LIGHTNESS_LIFT   // e.g. 0.12 at max
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = face.segmentationMask[y][x]
                if (maskVal < MASK_THRESHOLD) continue
                val idx = y * w + x
                val p = pixels[idx]
                val a = p ushr 24
                val r = p shr 16 and 0xFF
                val g = p shr 8  and 0xFF
                val b = p        and 0xFF

                // RGB → HSL
                val hsl = rgbToHsl(r, g, b)
                // Only lift L, clamp so we never exceed 1.0
                hsl[2] = (hsl[2] + liftFraction).coerceIn(0f, 1f)
                val rgb = hslToRgb(hsl[0], hsl[1], hsl[2])

                val enhanced = (a shl 24) or (rgb[0] shl 16) or (rgb[1] shl 8) or rgb[2]
                pixels[idx] = blendPixel(p, enhanced, maskVal * strength)
            }
        }
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Skin tone: unchanged from v1 — additive R/B nudge is fine.
     * Only tightened the max offsets slightly so it doesn't compound with the
     * HSL brightness change into an orange cast.
     */
    private fun applySkinTone(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val rOffset = (strength * 12f).toInt()   // was 18
        val bOffset = (strength * 10f).toInt()   // was 14
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = face.segmentationMask[y][x]
                if (maskVal < MASK_THRESHOLD) continue
                val idx = y * w + x
                val p = pixels[idx]
                val a = p ushr 24
                val r = ((p shr 16 and 0xFF) + rOffset).coerceIn(0, 255)
                val g = (p shr 8  and 0xFF)
                val b = ((p and 0xFF) - bOffset).coerceIn(0, 255)
                val enhanced = (a shl 24) or (r shl 16) or (g shl 8) or b
                pixels[idx] = blendPixel(p, enhanced, maskVal * strength)
            }
        }
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /** Blemish reduction: unchanged from v1 — logic is sound. */
    private fun applyBlemishReduction(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val blurred = gaussianBlur(pixels, w, h, BLEMISH_BLUR_RADIUS)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = face.segmentationMask[y][x]
                if (maskVal < MASK_THRESHOLD) continue
                val idx = y * w + x
                val p = pixels[idx]
                val bp = blurred[idx]
                val pLuma = luma(p); val bLuma = luma(bp)
                if (pLuma < bLuma - BLEMISH_DARK_THRESHOLD) {
                    pixels[idx] = blendPixel(p, bp, maskVal * strength * 0.8f)
                }
            }
        }
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /** Under-eye reduction: unchanged from v1 — ellipse feather is correct. */
    private fun applyUnderEyeReduction(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)

        listOf(face.leftEyeRect(w, h), face.rightEyeRect(w, h)).forEach { eyeRect ->
            val underRect = RectF(
                eyeRect.left, eyeRect.bottom,
                eyeRect.right, eyeRect.bottom + eyeRect.height() * 0.7f
            )
            val liftAmount = (strength * 60f).toInt().coerceIn(0, 255)
            for (y in underRect.top.toInt() until underRect.bottom.toInt().coerceAtMost(h)) {
                for (x in underRect.left.toInt() until underRect.right.toInt().coerceAtMost(w)) {
                    val idx = y * w + x
                    val p = pixels[idx]
                    val a = p ushr 24
                    val r = screen(p shr 16 and 0xFF, liftAmount)
                    val g = screen(p shr 8  and 0xFF, liftAmount)
                    val b = screen(p        and 0xFF, liftAmount)
                    val feather = ellipseFeather(underRect, x, y)
                    pixels[idx] = blendPixel(p, (a shl 24) or (r shl 16) or (g shl 8) or b,
                        strength * feather)
                }
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Eye brightness — now luminance-gated like teeth whitening.
     *
     * v1 brightened the entire eye bounding rect including eyelids and skin
     * above the eye.  Now only pixels that are already relatively bright
     * (whites of the eye, iris highlights) receive the lift, which is also
     * capped lower ([MAX_EYE_LIFT]) so the whites don't blow out.
     */
    private fun applyEyeBrightness(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)
        val liftAmount = (strength * MAX_EYE_LIFT).toInt().coerceIn(0, 255)

        listOf(face.leftEyeRect(w, h), face.rightEyeRect(w, h)).forEach { rect ->
            for (y in rect.top.toInt() until rect.bottom.toInt().coerceAtMost(h)) {
                for (x in rect.left.toInt() until rect.right.toInt().coerceAtMost(w)) {
                    val idx = y * w + x
                    val p = pixels[idx]
                    val luminance = luma(p)
                    // Only brighten pixels that already look like eye whites/iris
                    // (luminance > 80); skip dark pixels (eyelashes, eyelid skin)
                    val eyeLikeness = smoothstep(80f, 160f, luminance)
                    if (eyeLikeness <= 0f) continue

                    val a = p ushr 24
                    val r = screen(p shr 16 and 0xFF, liftAmount)
                    val g = screen(p shr 8  and 0xFF, liftAmount)
                    val b = screen(p        and 0xFF, liftAmount)
                    val feather = ellipseFeather(rect, x, y)
                    pixels[idx] = blendPixel(
                        p,
                        (a shl 24) or (r shl 16) or (g shl 8) or b,
                        strength * feather * eyeLikeness
                    )
                }
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /** Teeth whitening: unchanged from v1 — logic is already solid. */
    private fun applyTeethWhitening(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)
        val mouthRect = face.mouthRect(w, h)
        val liftAmount = (strength * 50f).toInt().coerceIn(0, 255)

        for (y in mouthRect.top.toInt() until mouthRect.bottom.toInt().coerceAtMost(h)) {
            for (x in mouthRect.left.toInt() until mouthRect.right.toInt().coerceAtMost(w)) {
                val idx = y * w + x
                val p = pixels[idx]
                val a = p ushr 24
                val r = p shr 16 and 0xFF
                val g = p shr 8  and 0xFF
                val b = p        and 0xFF
                val maxC = maxOf(r, g, b)
                val minC = minOf(r, g, b)
                val luminance = (r + g + b) / 3f
                val saturation = if (maxC == 0) 0f else (maxC - minC) / maxC.toFloat()
                val teethLikeness =
                    smoothstep(110f, 190f, luminance) * (1f - smoothstep(0.25f, 0.6f, saturation))
                if (teethLikeness <= 0f) continue

                val avg = (r + g + b) / 3
                val nr = (r + (avg - r) * strength * 0.5f).toInt().coerceIn(0, 255)
                val ng = (g + (avg - g) * strength * 0.5f).toInt().coerceIn(0, 255)
                val nb = (b + (avg - b) * strength * 0.5f).toInt().coerceIn(0, 255)
                val lifted = (a shl 24) or
                        (screen(nr, liftAmount) shl 16) or
                        (screen(ng, liftAmount) shl 8) or
                        screen(nb, liftAmount)
                val feather = ellipseFeather(mouthRect, x, y)
                pixels[idx] = blendPixel(p, lifted, strength * feather * teethLikeness)
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Unsharp-mask sharpening.
     *
     * Changes vs v1:
     * - Amount capped at [MAX_SHARPEN_AMOUNT] (0.65) — was 1.5, which caused
     *   halos and made pores look like craters.
     * - Only applied where maskVal > [SHARPEN_MASK_THRESHOLD] (0.6) to skip
     *   boundary pixels where halos are most visible.
     * - Runs first in the pipeline on the original unmodified image, so it
     *   doesn't amplify artefacts introduced by other effects.
     */
    private fun applyFaceSharpening(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val blurred = gaussianBlur(pixels, w, h, SHARPEN_BLUR_RADIUS)
        val box = face.boundingBox
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val out = IntArray(w * h)
        result.getPixels(out, 0, w, 0, 0, w, h)

        val amount = strength * MAX_SHARPEN_AMOUNT  // was strength * 1.5f

        for (y in box.top.toInt() until box.bottom.toInt().coerceAtMost(h)) {
            for (x in box.left.toInt() until box.right.toInt().coerceAtMost(w)) {
                val maskVal = face.segmentationMask.getOrNull(y)?.getOrNull(x) ?: 0f
                // Skip boundary pixels to avoid halos at mask edges
                if (maskVal < SHARPEN_MASK_THRESHOLD) continue

                val idx = y * w + x
                val p = pixels[idx]; val bp = blurred[idx]
                val a = p ushr 24
                val r = ((p shr 16 and 0xFF) + amount * ((p shr 16 and 0xFF) - (bp shr 16 and 0xFF))).toInt().coerceIn(0, 255)
                val g = ((p shr 8  and 0xFF) + amount * ((p shr 8  and 0xFF) - (bp shr 8  and 0xFF))).toInt().coerceIn(0, 255)
                val b = ((p        and 0xFF) + amount * ((p        and 0xFF) - (bp        and 0xFF))).toInt().coerceIn(0, 255)
                out[idx] = blendPixel(p, (a shl 24) or (r shl 16) or (g shl 8) or b, maskVal * strength)
            }
        }
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    // ── HSL colour space helpers ─────────────────────────────────────────────

    /**
     * RGB (0–255 each) → HSL (H in [0,360), S and L in [0,1]).
     */
    private fun rgbToHsl(r: Int, g: Int, b: Int): FloatArray {
        val rf = r / 255f; val gf = g / 255f; val bf = b / 255f
        val max = maxOf(rf, gf, bf); val min = minOf(rf, gf, bf)
        val l = (max + min) / 2f
        if (max == min) return floatArrayOf(0f, 0f, l) // achromatic
        val d = max - min
        val s = if (l > 0.5f) d / (2f - max - min) else d / (max + min)
        val h = when (max) {
            rf   -> ((gf - bf) / d + (if (gf < bf) 6f else 0f)) / 6f
            gf   -> ((bf - rf) / d + 2f) / 6f
            else -> ((rf - gf) / d + 4f) / 6f
        }
        return floatArrayOf(h * 360f, s, l)
    }

    /**
     * HSL (H in [0,360), S and L in [0,1]) → RGB (0–255 each).
     */
    private fun hslToRgb(h: Float, s: Float, l: Float): IntArray {
        if (s == 0f) {
            val v = (l * 255).toInt().coerceIn(0, 255)
            return intArrayOf(v, v, v)
        }
        val q = if (l < 0.5f) l * (1f + s) else l + s - l * s
        val p = 2f * l - q
        val hNorm = h / 360f
        fun hue2rgb(t0: Float): Int {
            var t = t0
            if (t < 0f) t += 1f; if (t > 1f) t -= 1f
            val v = when {
                t < 1f / 6f -> p + (q - p) * 6f * t
                t < 1f / 2f -> q
                t < 2f / 3f -> p + (q - p) * (2f / 3f - t) * 6f
                else         -> p
            }
            return (v * 255).toInt().coerceIn(0, 255)
        }
        return intArrayOf(
            hue2rgb(hNorm + 1f / 3f),
            hue2rgb(hNorm),
            hue2rgb(hNorm - 1f / 3f)
        )
    }

    // ── Low-level image math ─────────────────────────────────────────────────

    /**
     * Fast separable box-blur approximation of a Gaussian, operating on
     * packed ARGB int arrays without allocating intermediate Bitmaps.
     * Used only for blemish reduction and unsharp mask — NOT for skin
     * smoothing (which uses the bilateral filter above).
     */
    private fun gaussianBlur(src: IntArray, w: Int, h: Int, radius: Int): IntArray {
        if (radius <= 0) return src
        val tmp = IntArray(w * h)
        val out = IntArray(w * h)
        val div = 2 * radius + 1

        // Horizontal pass
        for (y in 0 until h) {
            var rSum = 0; var gSum = 0; var bSum = 0
            for (k in -radius..radius) {
                val px = src[y * w + k.coerceIn(0, w - 1)]
                rSum += px shr 16 and 0xFF
                gSum += px shr 8  and 0xFF
                bSum += px        and 0xFF
            }
            for (x in 0 until w) {
                tmp[y * w + x] = 0xFF000000.toInt() or
                        ((rSum / div) shl 16) or ((gSum / div) shl 8) or (bSum / div)
                val addX = (x + radius + 1).coerceIn(0, w - 1)
                val subX = (x - radius).coerceIn(0, w - 1)
                val add = src[y * w + addX]; val sub = src[y * w + subX]
                rSum += (add shr 16 and 0xFF) - (sub shr 16 and 0xFF)
                gSum += (add shr 8  and 0xFF) - (sub shr 8  and 0xFF)
                bSum += (add        and 0xFF) - (sub        and 0xFF)
            }
        }

        // Vertical pass
        for (x in 0 until w) {
            var rSum = 0; var gSum = 0; var bSum = 0
            for (k in -radius..radius) {
                val px = tmp[k.coerceIn(0, h - 1) * w + x]
                rSum += px shr 16 and 0xFF
                gSum += px shr 8  and 0xFF
                bSum += px        and 0xFF
            }
            for (y in 0 until h) {
                out[y * w + x] = 0xFF000000.toInt() or
                        ((rSum / div) shl 16) or ((gSum / div) shl 8) or (bSum / div)
                val addY = (y + radius + 1).coerceIn(0, h - 1)
                val subY = (y - radius).coerceIn(0, h - 1)
                val add = tmp[addY * w + x]; val sub = tmp[subY * w + x]
                rSum += (add shr 16 and 0xFF) - (sub shr 16 and 0xFF)
                gSum += (add shr 8  and 0xFF) - (sub shr 8  and 0xFF)
                bSum += (add        and 0xFF) - (sub        and 0xFF)
            }
        }
        return out
    }

    /** Linear interpolation between two packed ARGB pixels by [alpha]. */
    private fun blendPixel(a: Int, b: Int, alpha: Float): Int {
        val t = alpha.coerceIn(0f, 1f)
        val ar = a shr 16 and 0xFF; val br = b shr 16 and 0xFF
        val ag = a shr 8  and 0xFF; val bg = b shr 8  and 0xFF
        val ab = a        and 0xFF; val bb = b        and 0xFF
        val aa = a ushr 24
        return (aa shl 24) or
                (((ar + ((br - ar) * t).toInt()).coerceIn(0, 255)) shl 16) or
                (((ag + ((bg - ag) * t).toInt()).coerceIn(0, 255)) shl 8) or
                 ((ab + ((bb - ab) * t).toInt()).coerceIn(0, 255))
    }

    /** BT.601 luma of a packed ARGB pixel. */
    private fun luma(pixel: Int): Float {
        val r = pixel shr 16 and 0xFF
        val g = pixel shr 8  and 0xFF
        val b = pixel        and 0xFF
        return 0.299f * r + 0.587f * g + 0.114f * b
    }

    /**
     * Screen blend of a single 0-255 channel with a 0-255 light amount.
     * Used only for under-eye / teeth / eye-brightness screen lifts.
     */
    private fun screen(channel: Int, amount: Int): Int =
        (255 - ((255 - channel) * (255 - amount)) / 255).coerceIn(0, 255)

    /** Hermite smoothstep. */
    private fun smoothstep(edge0: Float, edge1: Float, value: Float): Float {
        if (edge0 == edge1) return if (value < edge0) 0f else 1f
        val t = ((value - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    /**
     * Elliptical feather: 1.0 at centre of [rect], smoothly → 0 at the edge.
     */
    private fun ellipseFeather(rect: RectF, x: Int, y: Int): Float {
        val cx = rect.centerX(); val cy = rect.centerY()
        val rx = (rect.width()  / 2f).coerceAtLeast(1f)
        val ry = (rect.height() / 2f).coerceAtLeast(1f)
        val nx = (x + 0.5f - cx) / rx
        val ny = (y + 0.5f - cy) / ry
        val dist = sqrt(nx * nx + ny * ny)
        return 1f - smoothstep(0.6f, 1f, dist)
    }

    companion object {
        // Skin smoothing
        private const val MASK_THRESHOLD            = 0.3f
        private const val MAX_BILATERAL_RADIUS      = 6        // was Gaussian 12 — much more conservative
        private const val BILATERAL_SPATIAL_SIGMA   = 3f       // controls spatial falloff
        private const val BILATERAL_COLOR_SIGMA     = 30f      // lower = more edge-preserving
        private const val TEXTURE_PRESERVE_AMOUNT   = 0.28f    // HF detail blended back after bilateral

        // Brightness
        private const val MAX_LIGHTNESS_LIFT        = 0.12f    // max HSL L raise (≈ 30/255)

        // Eye brightness
        private const val MAX_EYE_LIFT              = 40f      // was 70 — prevents whites blowing out

        // Blemish reduction (unchanged)
        private const val BLEMISH_BLUR_RADIUS       = 8
        private const val BLEMISH_DARK_THRESHOLD    = 20f

        // Sharpening
        private const val SHARPEN_BLUR_RADIUS       = 2
        private const val MAX_SHARPEN_AMOUNT        = 0.65f    // was 1.5 — eliminates halos
        private const val SHARPEN_MASK_THRESHOLD    = 0.6f     // skip low-confidence boundary pixels
    }
}
