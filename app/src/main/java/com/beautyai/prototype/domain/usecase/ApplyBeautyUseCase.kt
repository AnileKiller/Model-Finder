package com.beautyai.prototype.domain.usecase

import android.graphics.Bitmap
import android.graphics.RectF
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
     * Brightness boost on skin pixels using a screen blend rather than a
     * hard multiply. Screen blend asymptotically approaches 255 instead of
     * clipping abruptly, so already-bright skin doesn't blow out into a flat
     * plastic-looking patch at high strength.
     */
    private fun applySkinBrightness(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        // How strongly we "screen" toward white — kept modest so max strength
        // still preserves visible skin texture.
        val liftAmount = (strength * 90f).toInt().coerceIn(0, 255)
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
                val r = screen(p shr 16 and 0xFF, liftAmount)
                val g = screen(p shr 8  and 0xFF, liftAmount)
                val b = screen(p        and 0xFF, liftAmount)
                val enhanced = (a shl 24) or (r shl 16) or (g shl 8) or b
                pixels[idx] = blendPixel(pixels[idx], enhanced, maskVal * strength)
            }
        }
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Skin tone enhancement: slightly warms skin by nudging red up and blue
     * down with small additive offsets (not multiplicative), so it doesn't
     * compound with other effects into channel clipping.
     */
    private fun applySkinTone(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val rOffset = (strength * 18f).toInt()
        val bOffset = (strength * 14f).toInt()
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
                    pixels[idx] = blendPixel(p, (a shl 24) or (r shl 16) or (g shl 8) or b, strength * feather)
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
        val liftAmount = (strength * 70f).toInt().coerceIn(0, 255)

        listOf(face.leftEyeRect(w, h), face.rightEyeRect(w, h)).forEach { rect ->
            for (y in rect.top.toInt() until rect.bottom.toInt().coerceAtMost(h)) {
                for (x in rect.left.toInt() until rect.right.toInt().coerceAtMost(w)) {
                    val idx = y * w + x
                    val p = pixels[idx]
                    val a = p ushr 24
                    val r = screen(p shr 16 and 0xFF, liftAmount)
                    val g = screen(p shr 8  and 0xFF, liftAmount)
                    val b = screen(p        and 0xFF, liftAmount)
                    val feather = ellipseFeather(rect, x, y)
                    pixels[idx] = blendPixel(p, (a shl 24) or (r shl 16) or (g shl 8) or b, strength * feather)
                }
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * Teeth whitening: desaturates and lightens only pixels that already look
     * teeth-like (bright, low-saturation), inside a feathered ellipse instead
     * of a hard rectangle. This avoids flattening lips/skin caught inside the
     * mouth bounding box into a solid grey block at high strength.
     */
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
                // Only affect pixels that are plausibly teeth: bright and
                // fairly low-saturation. This spares lips/skin/gaps from
                // being flattened toward grey.
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

    /**
     * Screen blend of a single 0-255 channel value with a 0-255 "light" amount.
     * Unlike a straight multiply, this asymptotically approaches 255 rather
     * than clipping hard, so brightening effects don't blow out already-light
     * pixels into a flat, textureless block.
     */
    private fun screen(channel: Int, amount: Int): Int =
        (255 - ((255 - channel) * (255 - amount)) / 255).coerceIn(0, 255)

    /**
     * Smooth 0→1 falloff between [edge0] and [edge1] (Hermite interpolation),
     * used to softly gate effects by luminance/saturation instead of a hard
     * cutoff.
     */
    private fun smoothstep(edge0: Float, edge1: Float, value: Float): Float {
        if (edge0 == edge1) return if (value < edge0) 0f else 1f
        val t = ((value - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    /**
     * Returns a 0-1 falloff for point ([x], [y]) based on its normalised
     * distance from the center of [rect], modeled as an ellipse: 1.0 at the
     * center, smoothly fading to 0 at and beyond the rect's edge. This turns
     * hard-edged rectangular effect regions into soft, natural-looking ones.
     */
    private fun ellipseFeather(rect: RectF, x: Int, y: Int): Float {
        val cx = rect.centerX(); val cy = rect.centerY()
        val rx = (rect.width() / 2f).coerceAtLeast(1f)
        val ry = (rect.height() / 2f).coerceAtLeast(1f)
        val nx = (x + 0.5f - cx) / rx
        val ny = (y + 0.5f - cy) / ry
        val dist = kotlin.math.sqrt(nx * nx + ny * ny)
        return 1f - smoothstep(0.6f, 1f, dist)
    }

    companion object {
        private const val MASK_THRESHOLD       = 0.3f
        private const val MAX_BLUR_RADIUS      = 12
        private const val BLEMISH_BLUR_RADIUS  = 8
        private const val BLEMISH_DARK_THRESHOLD = 20f
        private const val SHARPEN_BLUR_RADIUS  = 2
    }
}
