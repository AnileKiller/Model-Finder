package com.beautyai.prototype.domain.model

import android.graphics.Bitmap

/** Represents the current state of the beauty-processing pipeline. */
sealed class ProcessingState {
    /** No image has been selected yet. */
    object Idle : ProcessingState()

    /** Models are running inference; [progress] is in [0,1]. */
    data class Processing(val progress: Float = 0f, val message: String = "") : ProcessingState()

    /** Both original and enhanced bitmaps are ready to display. */
    data class Success(
        val original: Bitmap,
        val enhanced: Bitmap,
        val faceData: FaceData?
    ) : ProcessingState()

    /** Something went wrong. */
    data class Error(val message: String) : ProcessingState()
}
