package com.beautyai.prototype.domain.usecase

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import com.beautyai.prototype.domain.model.BeautyParameters
import com.beautyai.prototype.domain.model.FaceData
import kotlin.math.sqrt

class ApplyBeautyUseCase {

    operator fun invoke(
        source: Bitmap,
        faceData: FaceData,
        params: BeautyParameters
    ): Bitmap {
        val effective = params.withGlobalIntensity()
        var result = source.copy(Bitmap.Config.ARGB_8888, true)

        // 1. Punch geometric exclusion zones (eyes, eyebrows, lips, nostrils) into the raw
        //    segmentation mask so that structural facial features are never smoothed or corrected.
        val refinedMask = punchExclusionZones(faceData.segmentationMask, faceData, source.width, source.height)

        // 2. Pre-blur the refined mask to feather the punched-out holes and eliminate hard edges.
        val blurredMask = preBlurMask(refinedMask, source.width, source.height, 12)

        // Sharpening first — works on original, unmodified detail
        if (effective.faceSharpening > 0f)
            result = applyFaceSharpening(result, faceData, blurredMask, effective.faceSharpening)

        if (effective.skinSmoothing > 0f)
            result = applySkinSmoothing(result, blurredMask, effective.skinSmoothing)

        if (effective.blemishReduction > 0f)
            result = applyBlemishReduction(result, blurredMask, effective.blemishReduction)

        if (effective.skinBrightness > 0f)
            result = applySkinBrightness(result, blurredMask, effective.skinBrightness)

        if (effective.skinToneEnhancement > 0f)
            result = applySkinTone(result, blurredMask, effective.skinToneEnhancement)

        if (effective.underEyeReduction > 0f)
            result = applyUnderEyeReduction(result, faceData, effective.underEyeReduction)

        if (effective.eyeBrightness > 0f)
            result = applyEyeBrightness(result, faceData, effective.eyeBrightness)

        if (effective.teethWhitening > 0f)
            result = applyTeethWhitening(result, faceData, effective.teethWhitening)

        return result
    }

    // ── Individual beauty effects ────────────────────────────────────────────

    private fun applySkinSmoothing(src: Bitmap, mask: Array<FloatArray>, strength: Float): Bitmap {
        val radius = (strength * MAX_BILATERAL_RADIUS).toInt().coerceAtLeast(1)
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        val spatialWeights = FloatArray((radius + 1) * (radius + 1))
        for (dy in 0..radius) for (dx in 0..radius) {
            val dist2 = (dx * dx + dy * dy).toFloat()
            spatialWeights[dy * (radius + 1) + dx] =
                kotlin.math.exp(-dist2 / (2f * BILATERAL_SPATIAL_SIGMA * BILATERAL_SPATIAL_SIGMA))
        }

        val bilateral = IntArray(w * h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = mask[y][x] // Using the blurred mask
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

        val out = IntArray(w * h)
        for (i in pixels.indices) {
            val maskY = i / w; val maskX = i % w
            val maskVal = mask.getOrNull(maskY)?.getOrNull(maskX) ?: 0f
            if (maskVal < MASK_THRESHOLD) { out[i] = bilateral[i]; continue }

            val orig = pixels[i]; val bil = bilateral[i]
            val a = orig ushr 24
            val detailR = ((orig shr 16 and 0xFF) - (bil shr 16 and 0xFF))
            val detailG = ((orig shr 8  and 0xFF) - (bil shr 8  and 0xFF))
            val detailB = ((orig        and 0xFF) - (bil        and 0xFF))

            val textureStr = TEXTURE_PRESERVE_AMOUNT * strength 
            val r = ((bil shr 16 and 0xFF) + detailR * textureStr).toInt().coerceIn(0, 255)
            val g = ((bil shr 8  and 0xFF) + detailG * textureStr).toInt().coerceIn(0, 255)
            val b = ((bil        and 0xFF) + detailB * textureStr).toInt().coerceIn(0, 255)
            out[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }

        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(out, 0, w, 0, 0, w, h)
        return result
    }

    private fun applySkinBrightness(src: Bitmap, mask: Array<FloatArray>, strength: Float): Bitmap {
        val liftFraction = strength * MAX_LIGHTNESS_LIFT 
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = mask[y][x]
                if (maskVal < MASK_THRESHOLD) continue
                val idx = y * w + x
                val p = pixels[idx]
                val a = p ushr 24
                val r = p shr 16 and 0xFF
                val g = p shr 8  and 0xFF
                val b = p        and 0xFF

                val hsl = rgbToHsl(r, g, b)
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

    private fun applySkinTone(src: Bitmap, mask: Array<FloatArray>, strength: Float): Bitmap {
        // 2. MODIFIED: Cut tone offsets in half to stop the "fake tan/muddy" look
        val rOffset = (strength * 6f).toInt()   
        val bOffset = (strength * 5f).toInt()   
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = mask[y][x]
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

    private fun applyBlemishReduction(src: Bitmap, mask: Array<FloatArray>, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        // I recommend lowering this radius in your companion object from 8 to 5. 
        // A radius of 8 pulls colors from too far away, which causes color bleeding near edges.
        val blurred = gaussianBlur(pixels, w, h, 5) 

        // The maximum luma difference. If a pixel is darker than this, it's an eye/lip edge, not a blemish.
        val MAX_BLEMISH_DIFFERENCE = 55f 

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = mask[y][x]
                if (maskVal < MASK_THRESHOLD) continue

                val idx = y * w + x
                val p = pixels[idx]
                val bp = blurred[idx]

                val pLuma = luma(p)
                val bLuma = luma(bp)
                val diff = bLuma - pLuma

                // 1. Check if it falls inside the "Blemish Zone" (Darker than skin, but not as dark as an eye)
                if (diff > BLEMISH_DARK_THRESHOLD && diff < MAX_BLEMISH_DIFFERENCE) {

                    // 2. Taper the effect near the limits to prevent harsh pixelated cutoffs
                    val edgeProtection = 1f - smoothstep(MAX_BLEMISH_DIFFERENCE - 15f, MAX_BLEMISH_DIFFERENCE, diff)

                    // Apply the fix, respecting the mask, user strength, and edge protection
                    val finalAlpha = maskVal * strength * 0.8f * edgeProtection

                    pixels[idx] = blendPixel(p, bp, finalAlpha)
                }
            }
        }

        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }


    private fun applyUnderEyeReduction(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)

        listOf(face.leftEyeRect(w, h), face.rightEyeRect(w, h)).forEach { eyeRect ->
            // 1. Expand the box to properly cover the inner tear trough and lower orbital bone
            val underRect = RectF(
                eyeRect.left - eyeRect.width() * 0.15f, 
                eyeRect.bottom - eyeRect.height() * 0.1f, // Overlap slightly to catch lashline shadows
                eyeRect.right + eyeRect.width() * 0.15f, 
                eyeRect.bottom + eyeRect.height() * 0.9f
            )

            // Lowered the global max lift slightly so it doesn't wash out
            val liftAmount = (strength * 45f).toInt().coerceIn(0, 255)

            for (y in underRect.top.toInt().coerceAtLeast(0) until underRect.bottom.toInt().coerceAtMost(h - 1)) {
                for (x in underRect.left.toInt().coerceAtLeast(0) until underRect.right.toInt().coerceAtMost(w - 1)) {
                    val idx = y * w + x
                    val p = pixels[idx]

                    val luminance = luma(p)

                    // 2. LUMINANCE GATING: Target ONLY the dark shadows.
                    // If luminance < 70 (dark shadow), shadowLikeness is 1.0 (Full effect).
                    // If luminance > 140 (bright cheekbone), shadowLikeness is 0.0 (No effect).
                    val shadowLikeness = 1f - smoothstep(70f, 140f, luminance)

                    // Skip pixels that are already bright to prevent the white "stroke" effect
                    if (shadowLikeness <= 0f) continue 

                    val a = p ushr 24

                    // 3. COLOR CORRECTION: Eye bags are usually blue/purple. 
                    // Pushing a pure screen blend makes them ashy/gray. 
                    // We lift Red slightly more than Blue to warm the shadow up and match natural skin.
                    val rLift = (liftAmount * 1.15f).toInt().coerceIn(0, 255)
                    val bLift = (liftAmount * 0.85f).toInt().coerceIn(0, 255)

                    val r = screen(p shr 16 and 0xFF, rLift)
                    val g = screen(p shr 8  and 0xFF, liftAmount)
                    val b = screen(p        and 0xFF, bLift)

                    val feather = ellipseFeather(underRect, x, y)

                    // Combine user strength, geometric feathering, and shadow targeting
                    val finalAlpha = strength * feather * shadowLikeness

                    pixels[idx] = blendPixel(p, (a shl 24) or (r shl 16) or (g shl 8) or b, finalAlpha)
                }
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

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

    private fun applyTeethWhitening(src: Bitmap, face: FaceData, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)
        val mouthRect = face.mouthRect(w, h)

        // Increased the max lift slightly for a cleaner white
        val liftAmount = (strength * 60f).toInt().coerceIn(0, 255)

        for (y in mouthRect.top.toInt().coerceAtLeast(0) until mouthRect.bottom.toInt().coerceAtMost(h - 1)) {
            for (x in mouthRect.left.toInt().coerceAtLeast(0) until mouthRect.right.toInt().coerceAtMost(w - 1)) {
                val idx = y * w + x
                val p = pixels[idx]
                val a = p ushr 24
                val r = p shr 16 and 0xFF
                val g = p shr 8  and 0xFF
                val b = p        and 0xFF

                // Using the existing luma helper function
                val luminance = luma(p) 

                // 1. THE NEW GATEKEEPER
                // Lips and gums have high Red and low Green. Yellow teeth have high Red AND high Green.
                val redness = (r - g).toFloat()

                // If luminance is high enough AND redness is low, it is a tooth.
                // This allows highly saturated yellow to pass, but strictly blocks pink/red lips.
                val teethLikeness = smoothstep(90f, 160f, luminance) * (1f - smoothstep(20f, 50f, redness))

                if (teethLikeness <= 0f) continue

                // 2. THE NEW WHITENING MATH
                // Find the brightest channel (usually Red or Green in stained teeth)
                val maxC = maxOf(r, maxOf(g, b)).toFloat()

                // Neutralize the yellow by boosting the weaker channels (especially Blue) up to the max channel
                val desatStrength = strength * 0.85f // Keep a tiny bit of natural warmth
                val nr = (r + (maxC - r) * desatStrength).toInt()
                val ng = (g + (maxC - g) * desatStrength).toInt()
                val nb = (b + (maxC - b) * desatStrength).toInt() // Blue gets the biggest boost, killing the yellow

                // Apply the screen blend for that glowing finish
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


    private fun applyFaceSharpening(src: Bitmap, face: FaceData, mask: Array<FloatArray>, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val blurred = gaussianBlur(pixels, w, h, SHARPEN_BLUR_RADIUS)
        val box = face.boundingBox
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val out = IntArray(w * h)
        result.getPixels(out, 0, w, 0, 0, w, h)

        val amount = strength * MAX_SHARPEN_AMOUNT 

        for (y in box.top.toInt() until box.bottom.toInt().coerceAtMost(h)) {
            for (x in box.left.toInt() until box.right.toInt().coerceAtMost(w)) {
                val maskVal = mask.getOrNull(y)?.getOrNull(x) ?: 0f
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

    private fun rgbToHsl(r: Int, g: Int, b: Int): FloatArray {
        val rf = r / 255f; val gf = g / 255f; val bf = b / 255f
        val max = maxOf(rf, gf, bf); val min = minOf(rf, gf, bf)
        val l = (max + min) / 2f
        if (max == min) return floatArrayOf(0f, 0f, l) 
        val d = max - min
        val s = if (l > 0.5f) d / (2f - max - min) else d / (max + min)
        val h = when (max) {
            rf   -> ((gf - bf) / d + (if (gf < bf) 6f else 0f)) / 6f
            gf   -> ((bf - rf) / d + 2f) / 6f
            else -> ((rf - gf) / d + 4f) / 6f
        }
        return floatArrayOf(h * 360f, s, l)
    }

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
     * 3. NEW: Fast separable box blur to feather the 2D FloatArray mask.
     */
    private fun preBlurMask(mask: Array<FloatArray>, w: Int, h: Int, radius: Int): Array<FloatArray> {
        if (radius <= 0) return mask
        val out = Array(h) { FloatArray(w) }
        val tmp = Array(h) { FloatArray(w) }
        val div = (2 * radius + 1).toFloat()

        // Horizontal pass
        for (y in 0 until h) {
            var sum = 0f
            for (k in -radius..radius) sum += mask[y][k.coerceIn(0, w - 1)]
            for (x in 0 until w) {
                tmp[y][x] = sum / div
                val addX = (x + radius + 1).coerceIn(0, w - 1)
                val subX = (x - radius).coerceIn(0, w - 1)
                sum += mask[y][addX] - mask[y][subX]
            }
        }
        // Vertical pass
        for (x in 0 until w) {
            var sum = 0f
            for (k in -radius..radius) sum += tmp[k.coerceIn(0, h - 1)][x]
            for (y in 0 until h) {
                out[y][x] = sum / div
                val addY = (y + radius + 1).coerceIn(0, h - 1)
                val subY = (y - radius).coerceIn(0, h - 1)
                sum += tmp[addY][x] - tmp[subY][x]
            }
        }
        return out
    }

    private fun gaussianBlur(src: IntArray, w: Int, h: Int, radius: Int): IntArray {
        if (radius <= 0) return src
        val tmp = IntArray(w * h)
        val out = IntArray(w * h)
        val div = 2 * radius + 1

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

    private fun luma(pixel: Int): Float {
        val r = pixel shr 16 and 0xFF
        val g = pixel shr 8  and 0xFF
        val b = pixel        and 0xFF
        return 0.299f * r + 0.587f * g + 0.114f * b
    }

    private fun screen(channel: Int, amount: Int): Int =
        (255 - ((255 - channel) * (255 - amount)) / 255).coerceIn(0, 255)

    private fun smoothstep(edge0: Float, edge1: Float, value: Float): Float {
        if (edge0 == edge1) return if (value < edge0) 0f else 1f
        val t = ((value - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun ellipseFeather(rect: RectF, x: Int, y: Int): Float {
        val cx = rect.centerX(); val cy = rect.centerY()
        val rx = (rect.width()  / 2f).coerceAtLeast(1f)
        val ry = (rect.height() / 2f).coerceAtLeast(1f)
        val nx = (x + 0.5f - cx) / rx
        val ny = (y + 0.5f - cy) / ry
        val dist = sqrt(nx * nx + ny * ny)
        // 4. MODIFIED: Changed from 0.6f to 0.0f to kill the pill-shaped halos
        return 1f - smoothstep(0.0f, 1f, dist)
    }

    // ── Debug overlay ────────────────────────────────────────────────────────

    /**
     * Renders a detailed colour-coded feature overlay for debugging.
     *
     * Layer order (bottom → top):
     *  1. **Green tint** — pixels inside the active skin zone the beauty pipeline touches.
     *  2. **Coloured filled polygons** — one colour per facial feature group, drawn
     *     using the 468 FaceMesh landmarks so each region is geometrically accurate:
     *       • Face oval  — white stroke (no fill)
     *       • Nose bridge  — light blue fill
     *       • Nose base/nostrils — purple fill
     *       • Lips outer — hot-pink fill
     *       • Mouth interior/teeth — yellow fill
     *       • Eyebrows — orange fill
     *       • Eyes — cyan fill
     *  3. **White dots** — all 468 landmark positions.
     *  4. **Legend panel** — colour key in the top-left corner.
     */
    fun renderMaskDebugOverlay(source: Bitmap, faceData: FaceData): Bitmap {
        val w  = source.width
        val h  = source.height
        val lm = faceData.landmarks

        // ── 1. Skin mask: green tint per-pixel ───────────────────────────────
        val refinedMask = punchExclusionZones(faceData.segmentationMask, faceData, w, h)
        val pixels      = IntArray(w * h)
        source.getPixels(pixels, 0, w, 0, 0, w, h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx     = y * w + x
                val refined = refinedMask[y][x]
                val p       = pixels[idx]
                if (refined >= MASK_THRESHOLD) {
                    val t  = (refined * 0.35f).coerceIn(0f, 1f)
                    val a  = p ushr 24
                    val r  = p shr 16 and 0xFF
                    val g  = p shr 8  and 0xFF
                    val b  = p        and 0xFF
                    pixels[idx] = (a shl 24) or
                        ((r + (0   - r) * t).toInt().coerceIn(0, 255) shl 16) or
                        ((g + (200 - g) * t).toInt().coerceIn(0, 255) shl 8)  or
                         (b + (80  - b) * t).toInt().coerceIn(0, 255)
                }
            }
        }
        val out = source.copy(Bitmap.Config.ARGB_8888, true)
        out.setPixels(pixels, 0, w, 0, 0, w, h)

        if (lm.size < 468) return out

        // ── 2. Feature polygons via Canvas ───────────────────────────────────
        val canvas = Canvas(out)

        fun landmarkPath(indices: List<Int>): Path {
            val path = Path()
            path.moveTo(lm[indices[0]].x * w, lm[indices[0]].y * h)
            for (i in 1 until indices.size) path.lineTo(lm[indices[i]].x * w, lm[indices[i]].y * h)
            path.close()
            return path
        }

        fun fillPaint(r: Int, g: Int, b: Int, alpha: Int = 130) = Paint().apply {
            color = Color.argb(alpha, r, g, b)
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        fun edgePaint(r: Int, g: Int, b: Int, sw: Float = 2.5f, alpha: Int = 230) = Paint().apply {
            color = Color.argb(alpha, r, g, b)
            style = Paint.Style.STROKE
            strokeWidth = sw
            isAntiAlias = true
        }

        // Face oval — white stroke, no fill (so skin tint shows through)
        canvas.drawPath(landmarkPath(FEATURE_FACE_OVAL),
            edgePaint(255, 255, 255, 1.5f, 140))

        // Nose bridge — light blue
        canvas.drawPath(landmarkPath(FEATURE_NOSE_BRIDGE), fillPaint(100, 181, 246, 140))
        canvas.drawPath(landmarkPath(FEATURE_NOSE_BRIDGE), edgePaint(130, 200, 255, 2f, 210))

        // Nose base / nostrils — purple
        canvas.drawPath(landmarkPath(FEATURE_NOSE_BASE), fillPaint(186, 104, 200, 140))
        canvas.drawPath(landmarkPath(FEATURE_NOSE_BASE), edgePaint(220, 150, 235, 2f, 220))

        // Outer lips — hot pink
        canvas.drawPath(landmarkPath(FEATURE_LIPS_OUTER), fillPaint(233, 30, 99, 140))
        canvas.drawPath(landmarkPath(FEATURE_LIPS_OUTER), edgePaint(255, 80, 130, 2.5f, 235))

        // Inner mouth / teeth — golden yellow (inner polygon drawn on top of lips fill)
        canvas.drawPath(landmarkPath(FEATURE_LIPS_INNER), fillPaint(255, 235, 59, 165))
        canvas.drawPath(landmarkPath(FEATURE_LIPS_INNER), edgePaint(255, 248, 110, 1.5f, 235))

        // Eyebrows — deep orange
        canvas.drawPath(landmarkPath(FEATURE_LEFT_BROW),  fillPaint(255, 152, 0, 150))
        canvas.drawPath(landmarkPath(FEATURE_RIGHT_BROW), fillPaint(255, 152, 0, 150))
        canvas.drawPath(landmarkPath(FEATURE_LEFT_BROW),  edgePaint(255, 190, 50, 2f, 235))
        canvas.drawPath(landmarkPath(FEATURE_RIGHT_BROW), edgePaint(255, 190, 50, 2f, 235))

        // Eyes — cyan (drawn last so they sit on top of brow fills at the corner)
        canvas.drawPath(landmarkPath(FEATURE_LEFT_EYE),  fillPaint(0, 200, 220, 155))
        canvas.drawPath(landmarkPath(FEATURE_RIGHT_EYE), fillPaint(0, 200, 220, 155))
        canvas.drawPath(landmarkPath(FEATURE_LEFT_EYE),  edgePaint(0, 240, 255, 2.5f, 245))
        canvas.drawPath(landmarkPath(FEATURE_RIGHT_EYE), edgePaint(0, 240, 255, 2.5f, 245))

        // ── 3. All 468 landmark dots ─────────────────────────────────────────
        val dotPaint  = Paint().apply {
            color = Color.argb(210, 255, 255, 255)
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val dotR = (w * 0.0025f).coerceIn(1.5f, 4f)
        for (l in lm) canvas.drawCircle(l.x * w, l.y * h, dotR, dotPaint)

        // ── 4. Legend ────────────────────────────────────────────────────────
        drawDebugLegend(canvas, w, h)

        return out
    }

    private fun drawDebugLegend(canvas: Canvas, imgW: Int, imgH: Int) {
        val entries = listOf(
            Color.argb(220, 0,   200, 80)  to "Skin (smoothed)",
            Color.argb(220, 0,   200, 220) to "Eyes",
            Color.argb(220, 255, 152, 0)   to "Eyebrows",
            Color.argb(220, 233, 30,  99)  to "Lips",
            Color.argb(220, 255, 235, 59)  to "Mouth / teeth",
            Color.argb(220, 186, 104, 200) to "Nose base",
            Color.argb(220, 100, 181, 246) to "Nose bridge",
        )
        val pad    = imgW * 0.022f
        val boxSz  = imgW * 0.028f
        val textSz = imgW * 0.022f
        val lineH  = boxSz * 1.6f
        val panelW = imgW * 0.44f
        val panelH = pad + entries.size * lineH + pad * 0.6f

        // Dark semi-transparent background panel
        val bgPaint = Paint().apply { color = Color.argb(178, 0, 0, 0); isAntiAlias = true }
        canvas.drawRoundRect(
            RectF(pad * 0.4f, pad * 0.4f, pad * 0.4f + panelW, pad * 0.4f + panelH),
            14f, 14f, bgPaint
        )

        val boxPaint  = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
        val textPaint = Paint().apply {
            color     = Color.WHITE
            textSize  = textSz
            isAntiAlias = true
        }

        entries.forEachIndexed { i, (color, label) ->
            val top = pad * 0.4f + pad + i * lineH
            boxPaint.color = color
            canvas.drawRoundRect(RectF(pad, top, pad + boxSz, top + boxSz), 4f, 4f, boxPaint)
            canvas.drawText(label, pad + boxSz + pad * 0.5f, top + boxSz * 0.82f, textPaint)
        }
    }

    // ── Exclusion zone helpers ────────────────────────────────────────────────

    /**
     * Punches black "holes" into [baseMask] at the locations of eyes, eyebrows,
     * outer lips, and nostrils derived from the 468 FaceMesh landmarks.
     * Every pixel inside those polygons is set to 0.0 so that downstream
     * smoothing and blemish-reduction loops skip them instantly via the
     * MASK_THRESHOLD check — no extra per-pixel logic required.
     */
    private fun punchExclusionZones(
        baseMask: Array<FloatArray>,
        faceData: FaceData,
        w: Int,
        h: Int
    ): Array<FloatArray> {
        val maskBitmap = floatArrayToBitmap(baseMask, w, h)
        val canvas = Canvas(maskBitmap)

        val eraserPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        fun drawPolygon(indices: List<Int>) {
            if (indices.isEmpty()) return
            val path = Path()
            val first = faceData.landmarks[indices[0]]
            path.moveTo(first.x * w, first.y * h)
            for (i in 1 until indices.size) {
                val lm = faceData.landmarks[indices[i]]
                path.lineTo(lm.x * w, lm.y * h)
            }
            path.close()
            canvas.drawPath(path, eraserPaint)
        }

        // Left eye contour (MediaPipe canonical indices)
        drawPolygon(listOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246))

        // Right eye contour
        drawPolygon(listOf(362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398))

        // Left eyebrow
        drawPolygon(listOf(46, 53, 52, 65, 55, 107, 66, 105, 63, 70))

        // Right eyebrow
        drawPolygon(listOf(276, 283, 282, 295, 285, 336, 296, 334, 293, 300))

        // Outer lips
        drawPolygon(listOf(61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185))

        // Nostrils / nose base
        drawPolygon(listOf(4, 45, 51, 5, 281, 275, 440, 220))

        return bitmapToFloatArray(maskBitmap, w, h)
    }

    /**
     * Converts a 2-D FloatArray mask (values 0–1) into a greyscale ARGB_8888 Bitmap
     * so Android Canvas can draw exclusion polygons onto it.
     */
    private fun floatArrayToBitmap(mask: Array<FloatArray>, w: Int, h: Int): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val v = (mask[y][x].coerceIn(0f, 1f) * 255f).toInt()
                pixels[y * w + x] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
        }
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        return bmp
    }

    /**
     * Converts a greyscale ARGB_8888 Bitmap back into a 2-D FloatArray mask.
     * Uses the red channel (equal to green and blue in a greyscale image).
     */
    private fun bitmapToFloatArray(bmp: Bitmap, w: Int, h: Int): Array<FloatArray> {
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        return Array(h) { y ->
            FloatArray(w) { x ->
                ((pixels[y * w + x] shr 16) and 0xFF) / 255f
            }
        }
    }

    companion object {
        // ── Debug overlay: facial feature landmark index groups ───────────────
        // All indices follow the canonical MediaPipe FaceMesh 468-point topology.

        /** Outer face contour (36 points going clockwise from forehead). */
        private val FEATURE_FACE_OVAL = listOf(
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        )
        /** Left eye contour (16 points). */
        private val FEATURE_LEFT_EYE  = listOf(33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
        /** Right eye contour (16 points). */
        private val FEATURE_RIGHT_EYE = listOf(362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
        /** Left eyebrow (10 points). */
        private val FEATURE_LEFT_BROW = listOf(70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
        /** Right eyebrow (10 points). */
        private val FEATURE_RIGHT_BROW = listOf(300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
        /** Outer lip contour (20 points). */
        private val FEATURE_LIPS_OUTER = listOf(61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
        /** Inner lip / mouth-opening contour (20 points) — encloses the teeth area. */
        private val FEATURE_LIPS_INNER = listOf(78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
        /**
         * Nose bridge: a thin diamond from the nasal bridge (168) down through
         * left-side (44) and right-side (274) to the tip (4).  Gives the bridge
         * region a visible width while staying tight to the nose centreline.
         */
        private val FEATURE_NOSE_BRIDGE = listOf(168, 44, 4, 274)
        /** Nose base / nostril flare area (8 points). */
        private val FEATURE_NOSE_BASE   = listOf(4, 45, 51, 5, 281, 275, 440, 220)

        // Skin smoothing
        private const val MASK_THRESHOLD            = 0.3f
        private const val MAX_BILATERAL_RADIUS      = 6       
        private const val BILATERAL_SPATIAL_SIGMA   = 3f       
        private const val BILATERAL_COLOR_SIGMA     = 30f      
        private const val TEXTURE_PRESERVE_AMOUNT   = 0.28f    

        // Brightness
        private const val MAX_LIGHTNESS_LIFT        = 0.12f    

        // Eye brightness
        private const val MAX_EYE_LIFT              = 40f      

        // Blemish reduction
        private const val BLEMISH_BLUR_RADIUS       = 8
        private const val BLEMISH_DARK_THRESHOLD    = 20f

        // Sharpening
        private const val SHARPEN_BLUR_RADIUS       = 2
        private const val MAX_SHARPEN_AMOUNT        = 0.65f    
        private const val SHARPEN_MASK_THRESHOLD    = 0.6f     
    }
}
