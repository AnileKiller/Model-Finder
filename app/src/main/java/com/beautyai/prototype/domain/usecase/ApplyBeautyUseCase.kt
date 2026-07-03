package com.beautyai.prototype.domain.usecase

import android.graphics.Bitmap
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

        // 1. NEW: Pre-blur the segmentation mask to eliminate the "Phantom Mask" hard edges.
        // A radius of 8-12 pixels creates a smooth gradient falloff at the borders.
        val blurredMask = preBlurMask(faceData.segmentationMask, source.width, source.height, 12)

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
        val blurred = gaussianBlur(pixels, w, h, BLEMISH_BLUR_RADIUS)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = mask[y][x]
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

    companion object {
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
