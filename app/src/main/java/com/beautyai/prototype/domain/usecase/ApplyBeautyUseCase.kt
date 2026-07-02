package com.beautyai.prototype.domain.usecase

import android.graphics.Bitmap
import com.beautyai.prototype.domain.model.BeautyParameters
import com.beautyai.prototype.domain.model.FaceData

/**
 * Applies the full beauty-enhancement pipeline to [source] using the pre-computed
 * [faceData] and user-controlled [params].
 *
 * Each effect is implemented as a pure function that returns a new Bitmap,
 * making them individually testable and easy to transplant into another project.
 *
 * Processing runs on whatever coroutine dispatcher the caller provides
 * (use [Dispatchers.Default] for background work).
 */
class ApplyBeautyUseCase {

    operator fun invoke(
        source: Bitmap,
        faceData: FaceData,
        params: BeautyParameters
    ): Bitmap {
        val effective = params.withGlobalIntensity()
        var result = source.copy(Bitmap.Config.ARGB_8888, true)

        if (effective.skinSmoothing > 0f)
            result = applySkinSmoothing(result, faceData, effective.skinSmoothing)

        if (effective.skinBrightness > 0f)
            result = applySkinBrightness(result, faceData, effective.skinBrightness)

        if (effective.skinToneEnhancement > 0f)
            result = applySkinTone(result, faceData, effective.skinToneEnhancement)

        if (effective.blemishReduction > 0f)
            result = applyBlemishReduction(result, faceData, effective.blemishReduction)

        if (effective.underEyeReduction > 0f)
            result = applyUnderEyeReduction(result, faceData, effective.underEyeReduction)

        if (effective.eyeBrightness > 0f)
            result = applyEyeBrightness(result, faceData, effective.eyeBrightness)

        if (effective.teethWhitening > 0f)
            result = applyTeethWhitening(result, faceData, effective.teethWhitening)

        if (effective.faceSharpening > 0f)
            result = applyFaceSharpening(result, faceData, effective.faceSharpening)

        return result
    }

    // ── Individual beauty effects ────────────────────────────────────────────

    /**
     * Gaussian-blur-based skin smoothing.
     * Only pixels with a segmentation mask value above [MASK_THRESHOLD] are blurred.
     * The blur radius scales with [strength] in [0, 1].
     */
    private fun applySkinSmoothing(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val radius = (strength * MAX_BLUR_RADIUS).toInt().coerceAtLeast(1)
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val blurred = gaussianBlur(pixels, w, h, radius)
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val out = IntArray(w * h)
        result.getPixels(out, 0, w, 0, 0, w, h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = face.segmentationMask[y][x]
                if (maskVal < MASK_THRESHOLD) continue
                val idx = y * w + x
                out[idx] = blendPixel(out[idx], blurred[idx], maskVal * strength)
            }
        }
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Multiplicative brightness boost on skin pixels.
     * Uses a simple gamma-style lift rather than full HSL to keep it fast.
     */
    private fun applySkinBrightness(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val boost = 1f + strength * 0.4f
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
                val r = ((p shr 16 and 0xFF) * boost).toInt().coerceIn(0, 255)
                val g = ((p shr 8  and 0xFF) * boost).toInt().coerceIn(0, 255)
                val b = ((p        and 0xFF) * boost).toInt().coerceIn(0, 255)
                val enhanced = (a shl 24) or (r shl 16) or (g shl 8) or b
                pixels[idx] = blendPixel(pixels[idx], enhanced, maskVal * strength)
            }
        }
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Skin tone enhancement: slightly warms skin by boosting red and reducing blue.
     */
    private fun applySkinTone(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val rBoost = 1f + strength * 0.15f
        val bReduce = 1f - strength * 0.08f
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
                val r = ((p shr 16 and 0xFF) * rBoost).toInt().coerceIn(0, 255)
                val g = (p shr 8  and 0xFF)
                val b = ((p and 0xFF) * bReduce).toInt().coerceIn(0, 255)
                val enhanced = (a shl 24) or (r shl 16) or (g shl 8) or b
                pixels[idx] = blendPixel(pixels[idx], enhanced, maskVal * strength)
            }
        }
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Blemish reduction: identifies dark spots relative to the local mean
     * skin brightness and smooths them toward the surrounding skin tone.
     */
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
                // Treat pixels significantly darker than their surroundings as blemishes
                if (pLuma < bLuma - BLEMISH_DARK_THRESHOLD) {
                    pixels[idx] = blendPixel(p, bp, maskVal * strength * 0.8f)
                }
            }
        }
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Under-eye reduction: lightens the region below each eye using landmark positions.
     */
    private fun applyUnderEyeReduction(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)

        listOf(
            face.leftEyeRect(w, h),
            face.rightEyeRect(w, h)
        ).forEach { eyeRect ->
            // Expand downward to cover the under-eye bag area
            val underRect = android.graphics.RectF(
                eyeRect.left, eyeRect.bottom,
                eyeRect.right, eyeRect.bottom + eyeRect.height() * 0.7f
            )
            val boost = 1f + strength * 0.25f
            for (y in underRect.top.toInt() until underRect.bottom.toInt().coerceAtMost(h)) {
                for (x in underRect.left.toInt() until underRect.right.toInt().coerceAtMost(w)) {
                    val idx = y * w + x
                    val p = pixels[idx]
                    val a = p ushr 24
                    val r = ((p shr 16 and 0xFF) * boost).toInt().coerceIn(0, 255)
                    val g = ((p shr 8  and 0xFF) * boost).toInt().coerceIn(0, 255)
                    val b = ((p        and 0xFF) * boost).toInt().coerceIn(0, 255)
                    pixels[idx] = blendPixel(p, (a shl 24) or (r shl 16) or (g shl 8) or b, strength)
                }
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Eye brightness: boosts the brightness of each iris region to make eyes appear
     * larger and more awake.
     */
    private fun applyEyeBrightness(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)
        val boost = 1f + strength * 0.35f

        listOf(face.leftEyeRect(w, h), face.rightEyeRect(w, h)).forEach { rect ->
            for (y in rect.top.toInt() until rect.bottom.toInt().coerceAtMost(h)) {
                for (x in rect.left.toInt() until rect.right.toInt().coerceAtMost(w)) {
                    val idx = y * w + x
                    val p = pixels[idx]
                    val a = p ushr 24
                    val r = ((p shr 16 and 0xFF) * boost).toInt().coerceIn(0, 255)
                    val g = ((p shr 8  and 0xFF) * boost).toInt().coerceIn(0, 255)
                    val b = ((p        and 0xFF) * boost).toInt().coerceIn(0, 255)
                    pixels[idx] = blendPixel(p, (a shl 24) or (r shl 16) or (g shl 8) or b, strength)
                }
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Teeth whitening: desaturates and brightens the mouth region.
     */
    private fun applyTeethWhitening(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)
        val mouthRect = face.mouthRect(w, h)

        for (y in mouthRect.top.toInt() until mouthRect.bottom.toInt().coerceAtMost(h)) {
            for (x in mouthRect.left.toInt() until mouthRect.right.toInt().coerceAtMost(w)) {
                val idx = y * w + x
                val p = pixels[idx]
                val a = p ushr 24
                val r = p shr 16 and 0xFF
                val g = p shr 8  and 0xFF
                val b = p        and 0xFF
                val avg = (r + g + b) / 3
                // Blend toward grey (desaturate) then boost brightness
                val boostFactor = 1f + strength * 0.3f
                val nr = ((r + (avg - r) * strength) * boostFactor).toInt().coerceIn(0, 255)
                val ng = ((g + (avg - g) * strength) * boostFactor).toInt().coerceIn(0, 255)
                val nb = ((b + (avg - b) * strength) * boostFactor).toInt().coerceIn(0, 255)
                pixels[idx] = blendPixel(p, (a shl 24) or (nr shl 16) or (ng shl 8) or nb, strength)
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Unsharp-mask sharpening applied to the face bounding box region.
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

        for (y in box.top.toInt() until box.bottom.toInt().coerceAtMost(h)) {
            for (x in box.left.toInt() until box.right.toInt().coerceAtMost(w)) {
                val idx = y * w + x
                val maskVal = face.segmentationMask.getOrNull(y)?.getOrNull(x) ?: 0f
                if (maskVal < MASK_THRESHOLD * 0.5f) continue

                val p = pixels[idx]; val bp = blurred[idx]
                val a = p ushr 24
                val amount = strength * 1.5f
                val r = ((p shr 16 and 0xFF) + amount * ((p shr 16 and 0xFF) - (bp shr 16 and 0xFF))).toInt().coerceIn(0,255)
                val g = ((p shr 8  and 0xFF) + amount * ((p shr 8  and 0xFF) - (bp shr 8  and 0xFF))).toInt().coerceIn(0,255)
                val b = ((p        and 0xFF) + amount * ((p        and 0xFF) - (bp        and 0xFF))).toInt().coerceIn(0,255)
                out[idx] = blendPixel(p, (a shl 24) or (r shl 16) or (g shl 8) or b, maskVal * strength)
            }
        }
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    // ── Low-level image math ─────────────────────────────────────────────────

    /**
     * Fast separable Gaussian blur via two 1-D box-filter passes.
     * Operates on packed ARGB int arrays without allocating intermediate Bitmaps.
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
                val add = src[y * w + addX]
                val sub = src[y * w + subX]
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
                val add = tmp[addY * w + x]
                val sub = tmp[subY * w + x]
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
                (((ar + ((br - ar) * t).toInt()).coerceIn(0,255)) shl 16) or
                (((ag + ((bg - ag) * t).toInt()).coerceIn(0,255)) shl 8) or
                 ((ab + ((bb - ab) * t).toInt()).coerceIn(0,255))
    }

    /** BT.601 luma of a packed ARGB pixel. */
    private fun luma(pixel: Int): Float {
        val r = pixel shr 16 and 0xFF
        val g = pixel shr 8  and 0xFF
        val b = pixel        and 0xFF
        return 0.299f * r + 0.587f * g + 0.114f * b
    }

    companion object {
        private const val MASK_THRESHOLD       = 0.3f
        private const val MAX_BLUR_RADIUS      = 12
        private const val BLEMISH_BLUR_RADIUS  = 8
        private const val BLEMISH_DARK_THRESHOLD = 20f
        private const val SHARPEN_BLUR_RADIUS  = 2
    }
}
