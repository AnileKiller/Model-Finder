package com.beautyai.prototype.domain.model

import android.graphics.RectF

/**
 * Normalised 3-D landmark point (x, y in [0,1] relative to the input image;
 * z is relative depth — smaller values are closer to the camera).
 */
data class Landmark(val x: Float, val y: Float, val z: Float)

/**
 * All AI inference results for a single detected face, passed between
 * domain use-cases and the beauty pipeline.
 */
data class FaceData(
    /**
     * Bounding box returned by BlazeFace, in pixel coordinates relative to the
     * original source image dimensions.
     */
    val boundingBox: RectF,

    /**
     * 468 facial landmarks from FaceMesh in normalised [0,1] coordinates.
     * Index mapping follows the canonical MediaPipe FaceMesh topology:
     *   https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
     */
    val landmarks: List<Landmark>,

    /**
     * Per-pixel segmentation mask produced by the face segmentation model.
     * Float values are in [0,1]; 1 = skin, 0 = background.
     * Dimensions match the original source image.
     */
    val segmentationMask: Array<FloatArray>
) {
    companion object {
        // ── Landmark index constants ──────────────────────────────────────────
        // Left eye
        const val LEFT_EYE_TOP        = 159
        const val LEFT_EYE_BOTTOM     = 145
        const val LEFT_EYE_LEFT       = 33
        const val LEFT_EYE_RIGHT      = 133

        // Right eye
        const val RIGHT_EYE_TOP       = 386
        const val RIGHT_EYE_BOTTOM    = 374
        const val RIGHT_EYE_LEFT      = 362
        const val RIGHT_EYE_RIGHT     = 263

        // Under-eye
        const val LEFT_UNDER_EYE      = 253
        const val RIGHT_UNDER_EYE     = 23

        // Mouth corners
        const val MOUTH_LEFT          = 61
        const val MOUTH_RIGHT         = 291
        const val MOUTH_TOP           = 0
        const val MOUTH_BOTTOM        = 17
    }

    /** Pixel-space bounding rect for the left eye, derived from landmarks. */
    fun leftEyeRect(imageWidth: Int, imageHeight: Int): RectF = landmarkRect(
        indices = listOf(LEFT_EYE_LEFT, LEFT_EYE_RIGHT, LEFT_EYE_TOP, LEFT_EYE_BOTTOM),
        width = imageWidth, height = imageHeight, padding = 0.05f
    )

    /** Pixel-space bounding rect for the right eye, derived from landmarks. */
    fun rightEyeRect(imageWidth: Int, imageHeight: Int): RectF = landmarkRect(
        indices = listOf(RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
        width = imageWidth, height = imageHeight, padding = 0.05f
    )

    /** Pixel-space bounding rect for the mouth/teeth region. */
    fun mouthRect(imageWidth: Int, imageHeight: Int): RectF = landmarkRect(
        indices = listOf(MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM),
        width = imageWidth, height = imageHeight, padding = 0.02f
    )

    private fun landmarkRect(
        indices: List<Int>,
        width: Int,
        height: Int,
        padding: Float
    ): RectF {
        val xs = indices.map { landmarks[it].x * width }
        val ys = indices.map { landmarks[it].y * height }
        val padX = (xs.max() - xs.min()) * padding
        val padY = (ys.max() - ys.min()) * padding
        return RectF(
            (xs.min() - padX).coerceAtLeast(0f),
            (ys.min() - padY).coerceAtLeast(0f),
            (xs.max() + padX).coerceAtMost(width.toFloat()),
            (ys.max() + padY).coerceAtMost(height.toFloat())
        )
    }
}
