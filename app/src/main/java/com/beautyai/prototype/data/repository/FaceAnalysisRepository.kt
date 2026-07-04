package com.beautyai.prototype.data.repository

import android.content.Context
import android.graphics.Bitmap
import com.beautyai.prototype.data.inference.FaceLandmarkerDetector
import com.beautyai.prototype.data.inference.FaceSegmentationDetector
import com.beautyai.prototype.domain.model.FaceData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Orchestrates the two-stage face analysis pipeline:
 *
 *  1. **FaceLandmarker** (MediaPipe Tasks) — runs face detection and 478-point
 *     landmark extraction in a single call.  This replaces the former
 *     BlazeFace + FaceMesh two-step chain.  The Tasks SDK bundles the custom
 *     C++ ops (Landmarks2TransformMatrix, etc.) required by the Attention model,
 *     which the standard TFLite runtime cannot provide.
 *
 *  2. **FaceSegmentationDetector** (raw TFLite) — generates the per-pixel skin
 *     mask used by the beauty effects pipeline.  Unchanged from before.
 *
 * All inference runs on [Dispatchers.Default] so it never blocks the UI thread.
 * Detectors are initialised lazily on first use and kept alive for the lifetime
 * of the repository — call [close] when the owning ViewModel is cleared.
 */
class FaceAnalysisRepository(private val context: Context) : AutoCloseable {

    private val landmarker by lazy { FaceLandmarkerDetector(context) }
    private val segmenter  by lazy { FaceSegmentationDetector(context) }

    /**
     * Runs the full two-stage pipeline on [bitmap].
     *
     * @return [FaceData] if a face was detected, or null if no face is found.
     */
    suspend fun analyse(
        bitmap: Bitmap,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): FaceData? = withContext(Dispatchers.Default) {

        onProgress(0.1f, "Detecting face + landmarks…")
        val (box, landmarks) = landmarker.detect(bitmap) ?: return@withContext null

        onProgress(0.7f, "Generating skin mask…")
        val mask = segmenter.segment(bitmap, box)

        onProgress(1.0f, "Done")
        FaceData(boundingBox = box, landmarks = landmarks, segmentationMask = mask)
    }

    override fun close() {
        runCatching { landmarker.close() }
        runCatching { segmenter.close() }
    }
}
