package com.beautyai.prototype.data.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import com.beautyai.prototype.domain.model.Landmark
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions
/**
 * Wraps the MediaPipe Tasks [FaceLandmarker] to perform face detection and
 * facial landmark extraction in a single call.
 *
 * Model:  face_landmarker.task  (MediaPipe Tasks bundle)
 *   This is a composite bundle that contains:
 *     - BlazeFace short-range detector  (face detection)
 *     - face_landmark_with_attention    (478-point landmark extraction)
 *   Download from:
 *     https://storage.googleapis.com/mediapipe-models/face_landmarker/
 *         face_landmarker/float16/latest/face_landmarker.task
 *   Place at:  app/src/main/assets/models/face_landmarker.task
 *
 * Why MediaPipe Tasks and not raw TFLite:
 *   face_landmark_with_attention.tflite uses the custom C++ op
 *   [Landmarks2TransformMatrix] that is NOT present in the standard TFLite
 *   runtime. The Tasks SDK ships all required custom ops pre-compiled, so the
 *   model runs without any extra native code.
 *
 * Output:
 *   478 normalised landmarks — 468 standard face points (same indices as the
 *   legacy FaceMesh topology) plus 5 left-iris + 5 right-iris points.
 *   All coordinates are in [0,1] relative to the input bitmap. Downstream code
 *   that accesses indices 0-467 is fully compatible with no changes needed.
 *
 * Acceleration fallback:
 *   GPU delegate is attempted first; if initialisation throws (e.g. unsupported
 *   driver), the detector re-initialises on CPU automatically.
 *
 * Thread safety: NOT thread-safe. Create one instance per coroutine scope or
 * guard calls with a mutex.
 */
class FaceLandmarkerDetector(context: Context) : AutoCloseable {

    private val faceLandmarker: FaceLandmarker

    init {
        // Try GPU first, fall back to CPU silently.
        faceLandmarker =
            buildLandmarker(context, Delegate.GPU)
                ?: buildLandmarker(context, Delegate.CPU)
                ?: error(
                    "FaceLandmarker failed to initialise on both GPU and CPU delegates. " +
                    "Ensure face_landmarker.task is present in assets/models/."
                )
    }

    /**
     * Runs face detection + landmark extraction on [bitmap].
     *
     * Returns a [Pair] of:
     *  - bounding box in **pixel coordinates** (derived from the convex hull of
     *    all detected landmarks — no separate BlazeFace call needed)
     *  - list of 478 [Landmark]s in normalised [0,1] image coordinates
     *
     * Returns `null` if no face is found above the confidence thresholds.
     */
    fun detect(bitmap: Bitmap): Pair<RectF, List<Landmark>>? {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result  = faceLandmarker.detect(mpImage)

        if (result.faceLandmarks().isEmpty()) return null

        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()

        val landmarks = result.faceLandmarks()[0].map { lm ->
            Landmark(x = lm.x(), y = lm.y(), z = lm.z())
        }

        // Derive the face bounding box from all landmark positions.
        val minX = (landmarks.minOf { it.x } * w).coerceAtLeast(0f)
        val minY = (landmarks.minOf { it.y } * h).coerceAtLeast(0f)
        val maxX = (landmarks.maxOf { it.x } * w).coerceAtMost(w)
        val maxY = (landmarks.maxOf { it.y } * h).coerceAtMost(h)

        return Pair(RectF(minX, minY, maxX, maxY), landmarks)
    }

    override fun close() = faceLandmarker.close()

    companion object {
        private const val MODEL_FILE = "models/face_landmarker.task"

        private fun buildOptions(delegate: Delegate): FaceLandmarker.FaceLandmarkerOptions {
            val base = BaseOptions.builder()
                .setModelAssetPath(MODEL_FILE)
                .setDelegate(delegate)
                .build()
            return FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()
        }

        /**
         * Tries to create a [FaceLandmarker] with [delegate].
         * Returns `null` (instead of throwing) so the caller can try a fallback.
         */
        private fun buildLandmarker(context: Context, delegate: Delegate): FaceLandmarker? =
            runCatching {
                FaceLandmarker.createFromOptions(context, buildOptions(delegate))
            }.getOrNull()
    }
}
