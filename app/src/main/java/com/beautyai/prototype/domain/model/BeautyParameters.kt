package com.beautyai.prototype.domain.model

/**
 * Holds all beauty enhancement slider values.
 * Each value is in the range [0f, 1f].
 * A value of 0 means no effect; 1 means maximum intensity.
 *
 * This class is the single source-of-truth for the pipeline and is designed
 * to be copy-pasted into other projects with zero modification.
 */
data class BeautyParameters(
    /** Multiplicative brightness boost applied to the skin region. */
    val skinBrightness: Float = 0.3f,

    /** Dark-spot detection and inpainting strength. */
    val blemishReduction: Float = 0.4f,

    /** Lighten the under-eye region identified by face landmarks. */
    val underEyeReduction: Float = 0.3f,

    /** Iris brightness enhancement using landmark-cropped eye regions. */
    val eyeBrightness: Float = 0.3f,

    /** Desaturate and lighten teeth detected in the mouth region. */
    val teethWhitening: Float = 0.3f,

    /** Unsharp-mask sharpening applied to the whole face. */
    val faceSharpening: Float = 0.3f,

    /** Darken and sharpen brow hair relative to local surroundings. */
    val eyebrowDefinition: Float = 0.3f,

    /** Global multiplier that scales all of the above effects. */
    val overallIntensity: Float = 0.5f
) {
    companion object {
        /** Returns parameters with all sliders at their defaults. */
        val DEFAULT = BeautyParameters()

        /** Returns parameters with all sliders zeroed (no processing). */
        val ZERO = BeautyParameters(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
    }

    /**
     * Returns a copy of these parameters with [overallIntensity] applied as
     * a global multiplier to every individual slider value.
     */
    fun withGlobalIntensity(): BeautyParameters = copy(
        skinBrightness        = skinBrightness        * overallIntensity,
        blemishReduction      = blemishReduction      * overallIntensity,
        underEyeReduction     = underEyeReduction     * overallIntensity,
        eyeBrightness         = eyeBrightness         * overallIntensity,
        teethWhitening        = teethWhitening        * overallIntensity,
        faceSharpening        = faceSharpening        * overallIntensity,
        eyebrowDefinition     = eyebrowDefinition     * overallIntensity
    )
}
