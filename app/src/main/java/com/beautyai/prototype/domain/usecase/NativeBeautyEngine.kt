package com.example.beautycamera.nativebridge

/**
 * JNI bridge to the native beauty-engine library (native-lib.cpp).
 *
 * All masks are 1D row-major FloatArray of size (width * height), values in [0, 1],
 * flattened from the Array<FloatArray> masks produced by ApplyBeautyUseCase.
 *
 * The pixel buffer is ARGB_8888 as produced by Bitmap.getPixels().
 */
object NativeBeautyEngine {

    init {
        System.loadLibrary("beauty-engine")
    }

    /**
     * Runs the full beauty pipeline natively and returns a new processed pixel array.
     *
     * @param pixels          Original ARGB_8888 pixels, size = width * height. Not modified.
     * @param width           Image width in pixels.
     * @param height          Image height in pixels.
     * @param refinedMask     Skin segmentation mask with exclusion zones punched out.
     * @param sharpMask       Pre-blurred face-oval mask used for sharpening.
     * @param blemishMask     Blemish-repair region mask (ocular shield applied).
     * @param brightnessMask  Skin-brightness region mask.
     * @param eyebrowMask     Eyebrow feature mask.
     * @param eyesMask        Eyes feature mask (used to protect eyes during under-eye work).
     * @param eyeBagsMask     Under-eye bags feature mask.
     * @param irisMask        Feathered iris/pupil mask.
     * @param teethMask       Feathered teeth mask.
     *
     * Strengths are all in [0, 1]; pass 0 to skip an effect.
     */
    external fun processImage(
        pixels: IntArray,
        width: Int,
        height: Int,
        refinedMask: FloatArray,
        sharpMask: FloatArray,
        blemishMask: FloatArray,
        brightnessMask: FloatArray,
        eyebrowMask: FloatArray,
        eyesMask: FloatArray,
        eyeBagsMask: FloatArray,
        irisMask: FloatArray,
        teethMask: FloatArray,
        blemishStrength: Float,
        sharpenStrength: Float,
        eyebrowStrength: Float,
        skinBrightnessStrength: Float,
        underEyeStrength: Float,
        eyeBrightnessStrength: Float,
        teethWhiteningStrength: Float
    ): IntArray
}
