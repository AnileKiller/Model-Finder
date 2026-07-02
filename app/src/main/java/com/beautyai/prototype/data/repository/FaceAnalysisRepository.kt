package com.beautyai.prototype.data.repository

import android.content.Context
import android.graphics.Bitmap
import com.beautyai.prototype.data.inference.BlazeFaceDetector
import com.beautyai.prototype.data.inference.FaceMeshDetector
import com.beautyai.prototype.data.inference.FaceSegmentationDetector
import com.beautyai.prototype.domain.model.FaceData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Orchestrates the three-stage face analysis pipeline:
 *   1. BlazeFace  — face detection (bounding box)
 *   2. FaceMesh   — 468 landmark extraction
 *   3. Segmentation — per-pixel skin mask
 *
 * All inference runs on [Dispatchers.Default] so it never blocks the UI thread.
 * The detectors are created lazily on first use and kept alive for the lifetime
 * of the repository — call [close] when the owning ViewModel is cleared.
 */
class FaceAnalysisRepository(private val context: Context) : AutoCloseable {

    // Lazy initialisation: models load on first inference call, not at startup.
    private val blazeFace  by lazy { BlazeFaceDetector(context) }
    private val faceMesh   by lazy { FaceMeshDetector(context) }
    private val segmenter  by lazy { FaceSegmentationDetector(context) }

    /**
     * Runs the full three-stage pipeline on [bitmap].
     *
     * @return [FaceData] if a face was detected, or null if no face is found.
     */
    suspend fun analyse(
        bitmap: Bitmap,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): FaceData? = withContext(Dispatchers.Default) {

        onProgress(0.1f, "Detecting face…")
        val box = blazeFace.detect(bitmap) ?: return@withContext null

        onProgress(0.4f, "Extracting landmarks…")
        val landmarks = faceMesh.detect(bitmap, box) ?: return@withContext null

        onProgress(0.7f, "Generating skin mask…")
        val mask = segmenter.segment(bitmap, box)

        onProgress(1.0f, "Done")
        FaceData(boundingBox = box, landmarks = landmarks, segmentationMask = mask)
    }

    override fun close() {
        // Use runCatching so a failure in one close() doesn't prevent the others.
        runCatching { blazeFace.close() }
        runCatching { faceMesh.close() }
        runCatching { segmenter.close() }
    }
}
