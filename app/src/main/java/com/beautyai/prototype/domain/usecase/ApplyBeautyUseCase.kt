package com.beautyai.prototype.domain.usecase

import android.graphics.Bitmap
import android.graphics.BlurMaskFilter
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
        params: BeautyParameters,
        onDebugLog: ((String) -> Unit)? = null
    ): Bitmap {
        val effective = params.withGlobalIntensity()
        var result = source.copy(Bitmap.Config.ARGB_8888, true)

        // 1. Punch geometric exclusion zones (eyes, eyebrows, lips, nostrils) into the raw
        //    segmentation mask so that structural facial features are never smoothed or corrected.
        val refinedMask = punchExclusionZones(faceData.segmentationMask, faceData, source.width, source.height)

        // 2. Geometric face-oval mask (heavily feathered, blurRadius = 35f) —
        // multiplying the segmentation mask by this kills any body-skin false
        // positive that isn't near the face outline (e.g. a hand in frame),
        // since the segmentation model has no notion of "this is a hand."
        val faceOvalMask = createFeatureMask(
            faceData, listOf(FEATURE_FACE_OVAL), source.width, source.height, 35f
        )

        // 3a. Smooth mask — segmentation-based (includes neck/body-skin at the
        // reduced 0.4x strength baked into the detector) x the face-oval
        // geometric mask, then feathered. Used for effects that should still
        // reach past the jawline onto visible neck skin, just less strongly.
        val smoothMask = preBlurMask(
            multiplyMasks(refinedMask, faceOvalMask), source.width, source.height, 12
        )

        // ADD THIS — prevents bilateral filter from smearing under-eye skin
        val eyeBagExcludedSmoothMask = punchRegionsIntoMask(
            smoothMask, faceData,
            listOf(
                LEFT_EYE_BAG_TIER3, RIGHT_EYE_BAG_TIER3,
                LEFT_EYE_BAG_TIER2, RIGHT_EYE_BAG_TIER2,
                LEFT_EYE_BAG_INDICES, RIGHT_EYE_BAG_INDICES
            ),
            source.width, source.height
        )

        // 3b. Sharp mask — face oval only (no exclusion zones, just a soft geometric mask)
        // We do NOT punch exclusion zones because the blemish reduction logic itself 
        // uses redness/contrast detection to avoid eyes and lips naturally.
        val sharpMask = preBlurMask(faceOvalMask, source.width, source.height, 12)

        // Blemishes must be detected on the unsharpened image: sharpening turns pores
        // and compression texture into false positives. Unlike sharpMask, this mask has
        // segmentation and feature exclusions, so it is safe around eyes/lips/hair.
        val blemishMask = preBlurMask(
            multiplyMasks(refinedMask, faceOvalMask), source.width, source.height, 2
        )
        if (effective.blemishReduction > 0f)
            result = applyBlemishReduction(result, blemishMask, effective.blemishReduction, onDebugLog)

        if (effective.faceSharpening > 0f)
            result = applyFaceSharpening(result, faceData, sharpMask, effective.faceSharpening)

        if (effective.skinSmoothing > 0f)
            result = applySkinSmoothing(result, eyeBagExcludedSmoothMask, effective.skinSmoothing)

        if (effective.skinBrightness > 0f)
            result = applySkinBrightness(result, smoothMask, effective.skinBrightness)

        if (effective.skinToneEnhancement > 0f)
            result = applySkinTone(result, smoothMask, effective.skinToneEnhancement)

        // 1. MASK FEED FIX: three distinct, non-overlapping-in-purpose masks so
        // each effect only ever samples the mask it geometrically needs —
        // eliminates the mathematical collision where the eye-bag blur bled
        // into the eyeball/lash line and the eye-brightness glow leaked onto
        // the whole eye socket instead of just the iris.
        if (effective.underEyeReduction > 0f) {
            val eyesMask = createFeatureMask(
                faceData, listOf(LEFT_EYE_INDICES, RIGHT_EYE_INDICES),
                source.width, source.height, EYES_BLUR_RADIUS
            )

            // Combine all tiers for a massive, full-coverage containment zone
            val allBagTiers = listOf(
                LEFT_EYE_BAG_INDICES, RIGHT_EYE_BAG_INDICES,
                LEFT_EYE_BAG_TIER2, RIGHT_EYE_BAG_TIER2,
                LEFT_EYE_BAG_TIER3, RIGHT_EYE_BAG_TIER3
            )
            val eyeBagsMask = createFeatureMask(
                faceData, allBagTiers, source.width, source.height, 20f
            )

            result = applyUnderEyeReduction(result, eyeBagsMask, eyesMask, refinedMask, effective.underEyeReduction)
        }

        if (effective.eyeBrightness > 0f) {
            val irisMask = createFeatureMask(
                faceData, listOf(LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES),
                source.width, source.height, IRIS_BLUR_RADIUS
            )
            result = applyEyeBrightness(result, faceData, irisMask, effective.eyeBrightness)
        }

        if (effective.teethWhitening > 0f) {
            val teethMask = createFeatureMask(
                faceData, listOf(FEATURE_LIPS_INNER),
                source.width, source.height, 4f
            )
            result = applyTeethWhitening(result, faceData, teethMask, effective.teethWhitening)
        }

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

    /**
     * 3. FIX: The old implementation round-tripped every pixel through
     * rgbToHsl()/hslToRgb() (multiple divisions + branches per pixel), which
     * was the single biggest contributor to multi-second stalls on full-res
     * photos. Brightness lift needs no colour-space conversion at all — a
     * per-channel integer Screen blend produces the same "lift shadows,
     * preserve highlights" look and runs entirely in cheap integer ops.
     */
    private fun applySkinBrightness(src: Bitmap, mask: Array<FloatArray>, strength: Float): Bitmap {
        val liftAmount = (strength * MAX_BRIGHTNESS_LIFT).toInt().coerceIn(0, 255)
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

                // Fast Integer Screen Blending — no float math, no HSL conversion.
                val br = 255 - ((255 - r) * (255 - liftAmount)) / 255
                val bg = 255 - ((255 - g) * (255 - liftAmount)) / 255
                val bb = 255 - ((255 - b) * (255 - liftAmount)) / 255

                val enhanced = (a shl 24) or (br shl 16) or (bg shl 8) or bb
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

    /**
     * Commercial-style blemish reduction with frequency separation.
     *
     * This version treats blemishes as two separate problems:
     *
     * 1. Low frequency colour/tone defects — redness, purple/brown marks, yellow scabs, and broad
     *    acne-cluster colour patches. These are repaired by moving the blurred skin base toward a
     *    clean weighted donor base sampled from nearby non-suspect skin.
     * 2. High frequency texture defects — raised pimple centres, sharp dark pores/scabs, and harsh
     *    local contrast. These are reduced by attenuating only the defect detail layer, while keeping
     *    normal skin texture so the result does not become waxy.
     *
     * The key quality improvement over a normal blur/inpaint pass is that donor colour is built from
     * the low-frequency layer and suspected blemishes are excluded from the donor weights. The final
     * pixel is then reconstructed as repaired base + controlled high-frequency detail, which gives a
     * beauty-editor style correction rather than a smeared patch.
     */
    private fun applyBlemishReduction(
        src: Bitmap,
        mask: Array<FloatArray>,
        strength: Float,
        onDebugLog: ((String) -> Unit)? = null
    ): Bitmap {
        val w = src.width
        val h = src.height
        val original = IntArray(w * h)
        src.getPixels(original, 0, w, 0, 0, w, h)

        var facePixels = 0
        val redHistogram = IntArray(129) // R-G from -64..64 on valid skin
        for (y in 0 until h) for (x in 0 until w) {
            if (mask[y][x] > 0.50f) {
                facePixels++
                val p = original[y * w + x]
                val rg = ((p shr 16 and 255) - (p shr 8 and 255)).coerceIn(-64, 64)
                redHistogram[rg + 64]++
            }
        }
        if (facePixels == 0) return src
        val faceScale = sqrt(facePixels.toFloat()).coerceAtLeast(80f)

        // Median facial redness makes the detector adapt to complexion and white balance.
        var medianRg = 0
        var cumulative = 0
        val half = (facePixels + 1) / 2
        for (bin in redHistogram.indices) {
            cumulative += redHistogram[bin]
            if (cumulative >= half) {
                medianRg = bin - 64
                break
            }
        }

        // Three scales: texture, individual lesions, and diffuse acne clusters.
        val detailRadius = (faceScale * 0.0045f).toInt().coerceIn(1, 3)
        val localRadius = (faceScale * 0.018f).toInt().coerceIn(5, 12)
        val broadRadius = (faceScale * 0.055f).toInt().coerceIn(16, 36)
        
        // 1. MASSIVE SPREAD: Pushes the detected spots outward until they bridge together
        val growRadius = (faceScale * 0.065f).toInt().coerceIn(24, 48)
        
        // 2. EXTREME DONOR REACH: Because the repair mask now covers the whole cheek, 
        // the donor must search past the cheek to find clean skin near the ears and jawline.
        val donorRadius = (faceScale * 0.120f).toInt().coerceIn(48, 96)

        // Detection confidence is deliberately independent of the UI strength. Otherwise low
        // strength allows blemishes back into the donor pool and contaminates their replacement.
        val confidence = Array(h) { FloatArray(w) }
        var coreHits = 0

        for (y in 0 until h) for (x in 0 until w) {
            val maskValue = mask[y][x]
            if (maskValue < 0.08f) continue

            val i = y * w + x
            val p = original[i]
            val lp = localBase[i]
            val bp = broadBase[i]
            val dp = detailBase[i]

            val r = p shr 16 and 255
            val g = p shr 8 and 255
            val b = p and 255
            val lr = lp shr 16 and 255
            val lg = lp shr 8 and 255
            val lb = lp and 255
            val br = bp shr 16 and 255
            val bg = bp shr 8 and 255
            val bb = bp and 255

            val y0 = 0.299f * r + 0.587f * g + 0.114f * b
            val yl = 0.299f * lr + 0.587f * lg + 0.114f * lb
            val yb = 0.299f * br + 0.587f * bg + 0.114f * bb
            val yd = luma(dp)

            // Red residual at two scales catches both isolated pimples and dense clusters.
            val rg = (r - g).toFloat()
            val localRed = rg - (lr - lg)
            val broadRed = rg - (br - bg)
            val complexionRed = rg - medianRg
            val focalRedScore = maxOf(
                smoothstep(2.5f, 18f, localRed),
                smoothstep(4f, 24f, broadRed)
            )
            val complexionRedScore = smoothstep(12f, 38f, complexionRed)

            // Purple/brown inflammatory marks often have little R-G residual. Measure their
            // chroma displacement from the broad skin field instead of requiring pure redness.
            val purpleNow = (r + b) * 0.5f - g
            val purpleBroad = (br + bb) * 0.5f - bg
            val chromaShift = kotlin.math.abs((r - g) - (br - bg)) +
                0.55f * kotlin.math.abs((b - g) - (bb - bg))
            val purpleScore = smoothstep(5f, 24f, purpleNow - purpleBroad)
            val chromaScore = smoothstep(9f, 32f, chromaShift)

            val darkResidual = maxOf(yl - y0, yb - y0)
            val darkScore = smoothstep(6f, 25f, darkResidual)
            val textureScore = smoothstep(2.5f, 14f, kotlin.math.abs(y0 - yd))

            // Complexion-relative redness is only supporting evidence, never a standalone score;
            // this prevents naturally rosy cheeks or warm makeup from being flattened.
            val redScore = maxOf(
                focalRedScore,
                complexionRedScore * maxOf(focalRedScore, textureScore * 0.65f, darkScore * 0.55f)
            )

            // Darkness/compact contrast must also have inflammatory chroma. That intentionally
            // favors preserving neutral freckles and moles over removing every possible blackhead.
            val inflammatoryEvidence = maxOf(redScore, purpleScore, chromaScore * 0.70f)
            val darkInflamed = darkScore * inflammatoryEvidence
            val compactDefect = darkScore * textureScore * inflammatoryEvidence * 0.62f
            val score = maxOf(redScore, purpleScore * 0.82f, darkInflamed, compactDefect)

            confidence[y][x] = (score * maskValue).coerceIn(0f, 1f)
            if (score > 0.12f) coreHits++
        }

        // Grow from the detected core to cover the whole lesion, then feather the boundary.
        val grown = maxFilterMask(confidence, w, h, growRadius)
        // 3. HEAVY FEATHER: Smooths the massive grown mask so it blends organically 
        // into the excluded boundaries without leaving hard lines.
        val featherRadius = (faceScale * 0.035f).toInt().coerceIn(12, 24)
        val repairMask = preBlurMask(grown, w, h, featherRadius)


        // Build a donor only from pixels that are both skin and confidently clean. This is the
        // critical fix for dense acne: the lesion can no longer donate its own red colour.
        val cleanWeight = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            val skin = mask[y][x].coerceIn(0f, 1f)
            val rejected = maxOf(grown[y][x], confidence[y][x]).coerceIn(0f, 1f)
            cleanWeight[i] = if (skin > 0.08f) skin * (1f - rejected) * (1f - rejected) else 0f
        }
        // A wider clean-only pass is the fallback for dense clusters with too little nearby
        // support. Only its final emergency fallback contains unfiltered broad colour.
        // 4. INCREASE MAXIMUM FALLBACK REACH: Let the fallback donor actually escape the cluster
        val farDonor = cleanWeightedBlurTarget(
            detailBase, cleanWeight, broadBase, w, h, (donorRadius * 2).coerceAtMost(160)
        )

        val donor = cleanWeightedBlurTarget(
            detailBase, cleanWeight, farDonor, w, h, donorRadius
        )

        val out = original.copyOf()
        val effectStrength = strength.coerceIn(0f, 1f)
        var repairedPixels = 0
        var alphaSum = 0f

        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            val confidenceValue = repairMask[y][x].coerceIn(0f, 1f)
            val alpha = (confidenceValue * mask[y][x] * effectStrength)
                .coerceIn(0f, BLEMISH_MAX_ALPHA)
            if (alpha <= 0.015f) continue

            val o = original[i]
            val base = detailBase[i]
            val target = donor[i]

            val or = o shr 16 and 255
            val og = o shr 8 and 255
            val ob = o and 255
            val baser = base shr 16 and 255
            val baseg = base shr 8 and 255
            val baseb = base and 255
            val tr = target shr 16 and 255
            val tg = target shr 8 and 255
            val tb = target and 255

            // Retain more texture on weak detections and less at severe lesion cores.
            val textureKeep = (0.52f - 0.30f * confidenceValue).coerceIn(0.20f, 0.52f)
            val rr = (tr + (or - baser) * textureKeep).toInt().coerceIn(0, 255)
            val rg = (tg + (og - baseg) * textureKeep).toInt().coerceIn(0, 255)
            val rb = (tb + (ob - baseb) * textureKeep).toInt().coerceIn(0, 255)
            val repaired = (o and -0x1000000) or (rr shl 16) or (rg shl 8) or rb
            out[i] = blendPixel(o, repaired, alpha)
            repairedPixels++
            alphaSum += alpha
        }

        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(out, 0, w, 0, 0, w, h)
        onDebugLog?.invoke(
            "Blemish adaptive: cores=$coreHits repaired=$repairedPixels " +
                "meanAlpha=${if (repairedPixels > 0) alphaSum / repairedPixels else 0f} " +
                "medianRG=$medianRg radii=$detailRadius/$localRadius/$broadRadius/$donorRadius"
        )
        return result
    }
    /**
     * Fast annular colour average: box blur at [outerRadius] minus box blur at [innerRadius].
     * This approximates sampling the immediate surroundings of a blemish instead of averaging
     * the blemish pixel into its own replacement colour.
     */
    private fun annularBlur(src: IntArray, w: Int, h: Int, innerRadius: Int, outerRadius: Int): IntArray {
        val inner = gaussianBlur(src, w, h, innerRadius)
        val outer = gaussianBlur(src, w, h, outerRadius)
        val innerArea = (2 * innerRadius + 1) * (2 * innerRadius + 1)
        val outerArea = (2 * outerRadius + 1) * (2 * outerRadius + 1)
        val ringArea = (outerArea - innerArea).coerceAtLeast(1)
        val out = IntArray(w * h)

        for (i in src.indices) {
            val op = outer[i]; val ip = inner[i]
            val r = (((op shr 16 and 255) * outerArea - (ip shr 16 and 255) * innerArea) / ringArea).coerceIn(0, 255)
            val g = (((op shr 8 and 255) * outerArea - (ip shr 8 and 255) * innerArea) / ringArea).coerceIn(0, 255)
            val b = (((op and 255) * outerArea - (ip and 255) * innerArea) / ringArea).coerceIn(0, 255)
            out[i] = (src[i] and -0x1000000) or (r shl 16) or (g shl 8) or b
        }
        return out
    }

    private fun maxFilterMask(mask: Array<FloatArray>, w: Int, h: Int, radius: Int): Array<FloatArray> {
        if (radius <= 0) return mask
        val out = Array(h) { FloatArray(w) }
        for (y in 0 until h) {
            for (x in 0 until w) {
                var m = 0f
                for (dy in -radius..radius) {
                    val yy = (y + dy).coerceIn(0, h - 1)
                    for (dx in -radius..radius) {
                        val xx = (x + dx).coerceIn(0, w - 1)
                        if (mask[yy][xx] > m) m = mask[yy][xx]
                    }
                }
                out[y][x] = m
            }
        }
        return out
    }

    /**
     * Builds a skin-colour donor image by blurring only clean weighted pixels. If the clean weighted
     * blur has too little support at a pixel, falls back to the supplied target. This is effectively
     * a cheap, stable inpainting target for beauty retouching.
     */
    private fun cleanWeightedBlurTarget(
        src: IntArray,
        cleanWeight: FloatArray,
        fallback: IntArray,
        w: Int,
        h: Int,
        radius: Int
    ): IntArray {
        val wr = FloatArray(w * h)
        val wg = FloatArray(w * h)
        val wb = FloatArray(w * h)
        for (i in src.indices) {
            val weight = cleanWeight[i].coerceIn(0f, 1f)
            val p = src[i]
            wr[i] = (p shr 16 and 255) * weight
            wg[i] = (p shr 8 and 255) * weight
            wb[i] = (p and 255) * weight
        }

        val bw = boxBlurFloat(cleanWeight, w, h, radius)
        val br = boxBlurFloat(wr, w, h, radius)
        val bg = boxBlurFloat(wg, w, h, radius)
        val bb = boxBlurFloat(wb, w, h, radius)
        val out = IntArray(w * h)
        for (i in src.indices) {
            val weight = bw[i]
            if (weight > 0.035f) {
                val r = (br[i] / weight).toInt().coerceIn(0, 255)
                val g = (bg[i] / weight).toInt().coerceIn(0, 255)
                val b = (bb[i] / weight).toInt().coerceIn(0, 255)
                out[i] = (src[i] and -0x1000000) or (r shl 16) or (g shl 8) or b
            } else {
                out[i] = fallback[i]
            }
        }
        return out
    }

    private fun boxBlurFloat(src: FloatArray, w: Int, h: Int, radius: Int): FloatArray {
        if (radius <= 0) return src
        val tmp = FloatArray(w * h)
        val out = FloatArray(w * h)
        val div = (2 * radius + 1).toFloat()

        for (y in 0 until h) {
            var sum = 0f
            for (k in -radius..radius) sum += src[y * w + k.coerceIn(0, w - 1)]
            for (x in 0 until w) {
                tmp[y * w + x] = sum / div
                val addX = (x + radius + 1).coerceIn(0, w - 1)
                val subX = (x - radius).coerceIn(0, w - 1)
                sum += src[y * w + addX] - src[y * w + subX]
            }
        }

        for (x in 0 until w) {
            var sum = 0f
            for (k in -radius..radius) sum += tmp[k.coerceIn(0, h - 1) * w + x]
            for (y in 0 until h) {
                out[y * w + x] = sum / div
                val addY = (y + radius + 1).coerceIn(0, h - 1)
                val subY = (y - radius).coerceIn(0, h - 1)
                sum += tmp[addY * w + x] - tmp[subY * w + x]
            }
        }
        return out
    }

    /** Three separable box passes approximate a Gaussian; gaussianBlur itself is one box pass. */
    private fun gaussianApprox(src: IntArray, w: Int, h: Int, radius: Int): IntArray {
        var result = src
        repeat(3) { result = gaussianBlur(result, w, h, radius) }
        return result
    }


    private fun applyUnderEyeReduction(
        src: Bitmap,
        bagsMask: Array<FloatArray>,
        eyesMask: Array<FloatArray>,
        skinMask: Array<FloatArray>,
        strength: Float
    ): Bitmap {
        val w = src.width; val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        // 1. Constrain to skin and combine masks
        val rawMaskArray = Array(h) { y ->
            FloatArray(w) { x ->
                (bagsMask[y][x] - eyesMask[y][x]).coerceAtLeast(0f) * skinMask[y][x]
            }
        }

        // 2. Feather the mask edge gently
        val featheredRawMask = preBlurMask(rawMaskArray, w, h, 3)

        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = featheredRawMask[y][x]
                if (maskVal <= 0.01f) continue

                val idx = y * w + x
                val p   = pixels[idx]
                val lum = luma(p) 

                // 3. Adaptive Shadow Lifting: Darker pixels get max lift, highlights get 0.
                val shadowFactor = smoothstep(0f, 0.5f, 1f - (lum / 255f)) 

                val adaptiveLift = (strength * MAX_UNDER_EYE_LIFT * shadowFactor).toInt()
                if (adaptiveLift <= 0) continue

                val a = p ushr 24
                val nr = (p shr 16 and 0xFF) + adaptiveLift
                val ng = (p shr 8  and 0xFF) + adaptiveLift
                val nb = (p        and 0xFF) + adaptiveLift

                // 4. Preserve Skin Warmth: Pull the Red channel up slightly faster
                val warmRed = (nr + (adaptiveLift * 0.15f)).toInt().coerceIn(0, 255)

                val enhanced = (a shl 24) or (warmRed shl 16) or (ng.coerceIn(0, 255) shl 8) or nb.coerceIn(0, 255)

                pixels[idx] = blendPixel(p, enhanced, maskVal * strength * shadowFactor)
            }
        }
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * 1. IRIS & PUPIL DISTINGUISHING: [irisMask] is a feathered alpha mask
     * built from the canonical MediaPipe iris landmarks (passed in from
     * [invoke]) so the "sparkle" brightening is concentrated on the
     * iris/pupil itself rather than the whole eye socket — otherwise the
     * whole eyeball glows instead of just the pupil.
     */
    private fun applyEyeBrightness(src: Bitmap, face: FaceData, irisMask: Array<FloatArray>, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)
        val liftAmount = (strength * MAX_EYE_LIFT).toInt().coerceIn(0, 255)

        listOf(face.leftEyeRect(w, h), face.rightEyeRect(w, h)).forEach { rect ->
            for (y in rect.top.toInt().coerceAtLeast(0) until rect.bottom.toInt().coerceAtMost(h)) {
                for (x in rect.left.toInt().coerceAtLeast(0) until rect.right.toInt().coerceAtMost(w)) {
                    val irisVal = irisMask[y][x]
                    if (irisVal < MASK_THRESHOLD) continue

                    val idx = y * w + x
                    val p = pixels[idx]
                    val luminance = luma(p)
                    val eyeLikeness = smoothstep(80f, 160f, luminance)
                    if (eyeLikeness <= 0f) continue

                    val a = p ushr 24
                    val r = screen(p shr 16 and 0xFF, liftAmount)
                    val g = screen(p shr 8  and 0xFF, liftAmount)
                    val b = screen(p        and 0xFF, liftAmount)
                    pixels[idx] = blendPixel(
                        p,
                        (a shl 24) or (r shl 16) or (g shl 8) or b,
                        strength * irisVal * eyeLikeness
                    )
                }
            }
        }
        result.setPixels(pixels, 0, w, 0, 0, w, h)
        return result
    }

    /**
     * 1. MIGRATED TO GEOMETRIC MASK: previously relied on [ellipseFeather]
     * over the whole mouth bounding box, which faded out real teeth near the
     * rect edges and let the effect leak onto lips/gums whenever the mouth
     * wasn't perfectly centered in its rect. Now gated by [teethMask] — a
     * feathered mask built directly from the inner-lip contour — so the
     * effect only ever touches the actual mouth opening.
     */
    private fun applyTeethWhitening(src: Bitmap, face: FaceData, teethMask: Array<FloatArray>, strength: Float): Bitmap {
        val w = src.width; val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val pixels = IntArray(w * h)
        result.getPixels(pixels, 0, w, 0, 0, w, h)

        // Stronger max lift so the effect reads clearly on all subjects
        val liftAmount = (strength * 100f).toInt().coerceIn(0, 255)

        // 1. BOUNDING BOX REMOVED: the mouthRect could clip real teeth near
        // its edges. The feathered teethMask is now the sole gate, so it's
        // safe (and correct) to scan the whole image.
        for (y in 0 until h) {
            for (x in 0 until w) {
                val maskVal = teethMask[y][x]
                if (maskVal <= 0f) continue

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

                // RELAXED REDNESS GATE: widened to 50f-100f so severely
                // yellow/stained teeth aren't misclassified as lips.
                val teethLikeness = smoothstep(80f, 160f, luminance) * (1f - smoothstep(50f, 100f, redness))

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

                pixels[idx] = blendPixel(p, lifted, strength * maskVal * teethLikeness)
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

    // ── Feature mask generation ──────────────────────────────────────────────

    /**
     * 1. NEW: Builds a feathered per-pixel alpha mask (values in [0,1], sized
     * [h][w]) for one or more geometric landmark regions — e.g. the iris/pupil
     * rings or the crescent-shaped eye-bag contours.
     *
     * Each entry in [polygons] is a list of FaceMesh landmark indices; every
     * list is mapped from normalised [0,1] landmark space to absolute Bitmap
     * coordinates and drawn as a filled white polygon on a black Canvas. A
     * native [BlurMaskFilter] on the fill Paint feathers the polygon edges so
     * downstream effects blend smoothly instead of showing a hard cutout.
     *
     * The resulting Bitmap is opaque black outside the polygons and white
     * (blurred to grey at the edges) inside them, so the red channel alone
     * (equal to green/blue everywhere, and — since the fill colour is pure
     * white — numerically equal to the blurred alpha coverage at the edges)
     * can be read back directly as the isolated [0,1] mask value.
     */
    private fun createFeatureMask(
        faceData: FaceData,
        polygons: List<List<Int>>,
        w: Int,
        h: Int,
        blurRadius: Float,
        expansion: Float = 0f
    ): Array<FloatArray> {
        val maskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(maskBitmap)
        canvas.drawColor(Color.BLACK)

        val fillPaint = Paint().apply {
            color = Color.WHITE
            isAntiAlias = true
            if (expansion > 0f) {
                style = Paint.Style.FILL_AND_STROKE
                strokeWidth = expansion
                strokeJoin = Paint.Join.ROUND
            } else {
                style = Paint.Style.FILL
            }
            if (blurRadius > 0f) {
                maskFilter = BlurMaskFilter(blurRadius, BlurMaskFilter.Blur.NORMAL)
            }
        }

        polygons.forEach { indices ->
            if (indices.isEmpty()) return@forEach
            val path = Path()
            val first = faceData.landmarks[indices[0]]
            path.moveTo(first.x * w, first.y * h)
            for (i in 1 until indices.size) {
                val lm = faceData.landmarks[indices[i]]
                path.lineTo(lm.x * w, lm.y * h)
            }
            path.close()
            canvas.drawPath(path, fillPaint)
        }

        return bitmapToFloatArray(maskBitmap, w, h)
    }

    private fun createTieredEyeBagsMask(
        faceData: FaceData,
        w: Int, h: Int,
        blurRadius: Float
    ): Array<FloatArray> {
        val maskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(maskBitmap)
        canvas.drawColor(Color.BLACK)

        fun drawTier(polygons: List<List<Int>>, alphaValue: Int) {
            val paint = Paint().apply {
                color = Color.argb(alphaValue, 255, 255, 255)
                style = Paint.Style.FILL
                isAntiAlias = true
                if (blurRadius > 0f) {
                    maskFilter = BlurMaskFilter(blurRadius, BlurMaskFilter.Blur.NORMAL)
                }
            }
            polygons.forEach { indices ->
                if (indices.isEmpty()) return@forEach
                val path = Path()
                val first = faceData.landmarks[indices[0]]
                path.moveTo(first.x * w, first.y * h)
                for (i in 1 until indices.size) {
                    val lm = faceData.landmarks[indices[i]]
                    path.lineTo(lm.x * w, lm.y * h)
                }
                path.close()
                canvas.drawPath(path, paint)
            }
        }
        // Draw from largest/dimmest to smallest/brightest
        drawTier(listOf(LEFT_EYE_BAG_TIER3, RIGHT_EYE_BAG_TIER3), 75)   // ~30% intensity
        drawTier(listOf(LEFT_EYE_BAG_TIER2, RIGHT_EYE_BAG_TIER2), 150)  // ~60% intensity
        drawTier(listOf(LEFT_EYE_BAG_INDICES, RIGHT_EYE_BAG_INDICES), 255) // 100% intensity

        return bitmapToFloatArray(maskBitmap, w, h)
    }

    // ── Low-level image math ─────────────────────────────────────────────────

    /**
     * Per-pixel multiply of two same-sized masks (used to combine the
     * segmentation-based mask with a purely geometric mask).
     */
    private fun multiplyMasks(a: Array<FloatArray>, b: Array<FloatArray>): Array<FloatArray> {
        val h = a.size; val w = if (h > 0) a[0].size else 0
        return Array(h) { y -> FloatArray(w) { x -> a[y][x] * b[y][x] } }
    }

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

        // Eye bags — light pink/magenta (Tiers 3, 2, and 1)
        canvas.drawPath(landmarkPath(LEFT_EYE_BAG_TIER3), fillPaint(255, 105, 180, 50))
        canvas.drawPath(landmarkPath(RIGHT_EYE_BAG_TIER3), fillPaint(255, 105, 180, 50))
        canvas.drawPath(landmarkPath(LEFT_EYE_BAG_TIER2), fillPaint(255, 105, 180, 80))
        canvas.drawPath(landmarkPath(RIGHT_EYE_BAG_TIER2), fillPaint(255, 105, 180, 80))
        canvas.drawPath(landmarkPath(LEFT_EYE_BAG_INDICES), fillPaint(255, 105, 180, 140))
        canvas.drawPath(landmarkPath(RIGHT_EYE_BAG_INDICES), fillPaint(255, 105, 180, 140))
        canvas.drawPath(landmarkPath(LEFT_EYE_BAG_INDICES), edgePaint(255, 20, 147, 1.5f, 190))
        canvas.drawPath(landmarkPath(RIGHT_EYE_BAG_INDICES), edgePaint(255, 20, 147, 1.5f, 190))

        // Pupils / iris — cyan/white (needs the 478-point mesh with iris
        // refinement landmarks 468-477; skipped if unavailable).
        if (lm.size >= 478) {
            canvas.drawPath(landmarkPath(LEFT_IRIS_INDICES),  fillPaint(255, 255, 255, 210))
            canvas.drawPath(landmarkPath(RIGHT_IRIS_INDICES), fillPaint(255, 255, 255, 210))
            canvas.drawPath(landmarkPath(LEFT_IRIS_INDICES),  edgePaint(0, 240, 255, 1.5f, 245))
            canvas.drawPath(landmarkPath(RIGHT_IRIS_INDICES), edgePaint(0, 240, 255, 1.5f, 245))
        }

        // ── 3. All 478 landmark dots and readable cheek indices ───────────────
        val dotPaint  = Paint().apply {
            color = Color.argb(255, 255, 255, 255)
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        val textPaint = Paint().apply {
            color = Color.YELLOW
            textSize = (w * 0.01f).coerceIn(12f, 26f) // Shrunk text size
            isAntiAlias = true
            // Drop shadow removed to prevent smearing
        }
        val dotR = (w * 0.0025f).coerceIn(1.5f, 4f)

        // Reference points for our bounding box
        val noseTip = lm[4]
        val leftEye = lm[159]
        val rightEye = lm[386]
        val upperLip = lm[0]
        val eyeBaseline = minOf(leftEye.y, rightEye.y) + 0.03f

        for ((index, l) in lm.withIndex()) {
            val px = l.x * w
            val py = l.y * h
            canvas.drawCircle(px, py, dotR, dotPaint)

            // FILTER: Only draw text in the cheek areas to prevent a massive blob of text
            val isBelowEyes = l.y > eyeBaseline
            val isAboveLip = l.y < upperLip.y
            // Exclude the highly dense nose bridge/tip area
            val distToNoseX = kotlin.math.abs(l.x - noseTip.x)
            val isNotNose = distToNoseX > 0.06f 

            if (isBelowEyes && isAboveLip && isNotNose) {
                canvas.drawText(index.toString(), px + 3f, py - 3f, textPaint)
            }
        }

        // ── 4. Legend ────────────────────────────────────────────────────────
        drawDebugLegend(canvas, w, h)

        return out
    }

    private fun drawDebugLegend(canvas: Canvas, imgW: Int, imgH: Int) {
        val entries = listOf(
            Color.argb(220, 0,   200, 80)  to "Skin (smoothed)",
            Color.argb(220, 0,   200, 220) to "Eyes",
            Color.argb(220, 255, 255, 255) to "Pupils",
            Color.argb(220, 255, 20,  147) to "Eye Bags",
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
    private fun punchRegionsIntoMask(
        baseMask: Array<FloatArray>,
        faceData: FaceData,
        regionGroups: List<List<Int>>,
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
        regionGroups.forEach { indices ->
            if (indices.isEmpty()) return@forEach
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
        return bitmapToFloatArray(maskBitmap, w, h)
    }

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

        // Ocular Zone (Eyes and Eyebrows)
        drawPolygon(FEATURE_LEFT_EYE)
        drawPolygon(FEATURE_RIGHT_EYE)
        drawPolygon(FEATURE_LEFT_BROW)
        drawPolygon(FEATURE_RIGHT_BROW)

        // Lips
        drawPolygon(FEATURE_LIPS_OUTER)

        // Nostrils / nose base (Expanded)
        drawPolygon(FEATURE_NOSE_BASE)

        // Do not exclude the glabella here. A permanent "bindi zone" hole also hides
        // common forehead acne. If cosmetic-mark preservation is needed, detect that mark
        // explicitly and pass it as a separate optional exclusion mask.

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
        /** Nostrils / nose base — expanded to cover outer wings and bottom holes. */
        private val FEATURE_NOSE_BASE   = listOf(4, 45, 129, 98, 97, 2, 326, 327, 358, 275)

        /** Bindi / Glabella zone (Tight diamond centered between the eyebrows). */
        private val FEATURE_BINDI_ZONE  = listOf(8, 107, 168, 336)

        // ── Eyes (canonical MediaPipe eye contour indices) — used only for
        // mask subtraction, to carve the eyeball/lash line back out of the
        // heavily-blurred eye-bag mask.
        /** Left eye contour (16 points). */
        val LEFT_EYE_INDICES  = FEATURE_LEFT_EYE
        /** Right eye contour (16 points). */
        val RIGHT_EYE_INDICES = FEATURE_RIGHT_EYE

        // ── Iris & pupil (canonical MediaPipe iris refinement indices) ────────
        // Perimeter-only (center pupil dot omitted): including the center
        // point made path.close() fold inward into a star/bowtie shape
        // instead of a clean round pupil.
        /** Left iris ring — perimeter only: Right, Top, Left, Bottom. */
        private val LEFT_IRIS_INDICES = listOf(469, 470, 471, 472)
        /** Right iris ring — perimeter only: Right, Top, Left, Bottom. */
        private val RIGHT_IRIS_INDICES = listOf(474, 475, 476, 477)

        // ── Eye bags (canonical MediaPipe crescent contour indices) ───────────
        // Corrected Left Eye Bag (Lower lid outer->inner, then cheek inner->outer)
        private val LEFT_EYE_BAG_INDICES = listOf(
            33, 7, 163, 144, 145, 153, 154, 155, 133,
            112, 26, 22, 23, 24, 110, 25, 226, 130
        )

        // Corrected Right Eye Bag (Lower lid inner->outer, then cheek outer->inner)
        private val RIGHT_EYE_BAG_INDICES = listOf(
            362, 382, 381, 380, 374, 373, 390, 249, 263,
            359, 446, 255, 339, 254, 253, 252, 256, 341
        )

        // Tier 2: Connects the bottom of Tier 1 down to the middle cheek row
        private val LEFT_EYE_BAG_TIER2 = listOf(
            112, 26, 22, 23, 24, 110, 25, // Inner to Outer (Top edge)
            31, 228, 229, 230, 231, 232, 233  // Outer to Inner (Bottom edge)
        )
        private val RIGHT_EYE_BAG_TIER2 = listOf(
            341, 256, 252, 253, 254, 339, 255, // Inner to Outer (Top edge)
            261, 448, 449, 450, 451, 452, 453  // Outer to Inner (Bottom edge)
        )

        // Tier 3: Connects the bottom of Tier 2 down to the lowest cheek shadow boundary
        private val LEFT_EYE_BAG_TIER3 = listOf(
            233, 232, 231, 230, 229, 228, 31, // Inner to Outer (Top edge)
            111, 117, 118, 119, 120, 121, 128 // Outer to Inner (Bottom edge)
        )
        private val RIGHT_EYE_BAG_TIER3 = listOf(
            453, 452, 451, 450, 449, 448, 261, // Inner to Outer (Top edge)
            340, 346, 347, 348, 349, 350, 357  // Outer to Inner (Bottom edge)
        )

        // Skin smoothing
        private const val MASK_THRESHOLD            = 0.2f
        private const val BLEMISH_MAX_ALPHA         = 0.985f
        private const val BLEMISH_DIFFUSE_REDNESS_ALPHA = 0.42f
        private const val BLEMISH_LUMA_TEXTURE_KEEP = 0.74f
        private const val BLEMISH_CHROMA_TEXTURE_KEEP = 0.62f
        private const val MAX_BILATERAL_RADIUS      = 6       
        private const val BILATERAL_SPATIAL_SIGMA   = 3f       
        private const val BILATERAL_COLOR_SIGMA     = 30f      
        private const val TEXTURE_PRESERVE_AMOUNT   = 0.28f    

        // Brightness — max Screen-blend lift amount (0-255 integer scale)
        private const val MAX_BRIGHTNESS_LIFT       = 50f

        // Eye brightness
        private const val MAX_EYE_LIFT              = 40f      
        private const val IRIS_BLUR_RADIUS          = 4f

        // Under-eye / eye-bag reduction
        private const val MAX_UNDER_EYE_LIFT        = 60f
        private const val EYE_BAG_BLUR_RADIUS       = 30f
        private const val EYES_BLUR_RADIUS          = 2f

        // Sharpening
        private const val SHARPEN_BLUR_RADIUS       = 2
        private const val MAX_SHARPEN_AMOUNT        = 0.65f    
        private const val SHARPEN_MASK_THRESHOLD    = 0.6f     
    }
}
