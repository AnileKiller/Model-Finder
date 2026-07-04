package com.beautyai.prototype.data.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.RectF
import com.beautyai.prototype.domain.model.Landmark
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Wraps the MediaPipe Face Landmark with Attention TFLite model to extract
 * 468 facial landmarks.
 *
 * Model:  face_landmark_with_attention.tflite
 * Input:  [1, 192, 192, 3] — normalised float RGB in [0, 1]
 * Output:
 *   - landmarks  [1, 1, 1, 1404]  (468 × 3 values: x, y, z in [0, 192])
 *   - score      [1, 1, 1, 1]     (face presence confidence logit)
 *
 * Compared to the lite model this version uses a self-attention mechanism
 * that produces significantly more accurate landmark positions around the
 * eyes, eyebrows, and lips — the exact regions used for exclusion-zone
 * punching. The I/O tensor shapes and coordinate space are identical.
 *
 * Landmarks are returned in normalised [0,1] image coordinates relative
 * to the original source bitmap.
 */
class FaceMeshDetector(context: Context) : AutoCloseable {

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?

    private val inputSize = 192
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * FLOAT_BYTES)
            .also { it.order(ByteOrder.nativeOrder()) }

    // Flat output: 468 landmarks × (x, y, z)
    private val landmarksOut = Array(1) { Array(1) { Array(1) { FloatArray(468 * 3) } } }
    // Shape: [1, 1, 1, 1]  (face presence confidence)
    private val scoreOut     = Array(1) { Array(1) { Array(1) { FloatArray(1) } } }

    init {
        val modelBuffer = loadModelFile(context, MODEL_FILE)
        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                addDelegate(gpuDelegate)
            } else {
                gpuDelegate = null
                setNumThreads(4)
                setUseNNAPI(true)
            }
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    /**
     * Runs FaceMesh on the region of [bitmap] specified by [faceBox].
     * Returns 468 landmarks in normalised [0,1] coordinates relative to the
     * *original* [bitmap], or null if the face presence score is too low.
     *
     * The face crop is letterboxed (aspect-ratio-preserving padding) before
     * being fed to the model. This prevents the squish distortion that occurs
     * when a non-square crop is forcefully scaled to the model's 192×192 input.
     * Without letterboxing, the model finds landmarks on a geometrically wrong
     * face, and the back-mapped coordinates float in the wrong direction
     * depending on the photo's aspect ratio (portrait vs. landscape).
     */
    fun detect(bitmap: Bitmap, faceBox: RectF, presenceThreshold: Float = 0.5f): List<Landmark>? {
        // Clamp crop dimensions to image bounds
        val cropLeft = faceBox.left.toInt().coerceAtLeast(0)
        val cropTop  = faceBox.top.toInt().coerceAtLeast(0)
        val cropW    = faceBox.width().toInt().coerceAtMost(bitmap.width  - cropLeft)
        val cropH    = faceBox.height().toInt().coerceAtMost(bitmap.height - cropTop)

        // Scale factor that fits the crop into inputSize×inputSize with no squish
        val scale  = minOf(inputSize.toFloat() / cropW, inputSize.toFloat() / cropH)
        val scaledW = (cropW * scale).toInt().coerceAtLeast(1)
        val scaledH = (cropH * scale).toInt().coerceAtLeast(1)

        // Padding offsets (letterbox bars)
        val padX = (inputSize - scaledW) / 2f
        val padY = (inputSize - scaledH) / 2f

        val letterboxed = cropAndScaleLetterbox(bitmap, cropLeft, cropTop, cropW, cropH, scaledW, scaledH)
        fillInputBuffer(letterboxed)

        val outputMap = mapOf(0 to landmarksOut, 1 to scoreOut)
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        val presenceScore = sigmoid(scoreOut[0][0][0][0])
        if (presenceScore < presenceThreshold) return null

        val flat = landmarksOut[0][0][0]
        return List(468) { i ->
            val modelX = flat[i * 3]
            val modelY = flat[i * 3 + 1]
            val z      = flat[i * 3 + 2]

            // 1. Subtract letterbox padding to get position inside the scaled crop
            // 2. Divide by scale to get pixel position within the original crop
            val cropXPx = (modelX - padX) / scale
            val cropYPx = (modelY - padY) / scale

            // 3. Add bounding-box origin to anchor back to full-image space,
            //    then normalise to [0, 1] relative to the full image dimensions
            Landmark(
                x = (cropLeft + cropXPx) / bitmap.width,
                y = (cropTop  + cropYPx) / bitmap.height,
                z = z
            )
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Crops [src] to the given pixel region, scales it to [scaledW]×[scaledH]
     * (aspect-ratio-preserving), then centres it on a black [inputSize]×[inputSize]
     * canvas — producing a letterboxed input for the FaceMesh model.
     */
    private fun cropAndScaleLetterbox(
        src: Bitmap,
        cropLeft: Int, cropTop: Int, cropW: Int, cropH: Int,
        scaledW: Int, scaledH: Int
    ): Bitmap {
        val cropped  = Bitmap.createBitmap(src, cropLeft, cropTop, cropW, cropH)
        val scaled   = Bitmap.createScaledBitmap(cropped, scaledW, scaledH, true)

        // Black canvas — padding pixels stay 0 and are ignored by the model
        val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(letterboxed)
        canvas.drawBitmap(scaled, (inputSize - scaledW) / 2f, (inputSize - scaledH) / 2f, null)
        return letterboxed
    }

    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in pixels) {
            inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255f) // R
            inputBuffer.putFloat((pixel shr 8  and 0xFF) / 255f) // G
            inputBuffer.putFloat((pixel        and 0xFF) / 255f) // B
        }
    }

    private fun sigmoid(x: Float) = 1f / (1f + Math.exp(-x.toDouble()).toFloat())

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    companion object {
        private const val MODEL_FILE = "models/face_landmark_with_attention.tflite"
        private const val FLOAT_BYTES = 4

        private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
            val assetFd = context.assets.openFd(filename)
            return FileInputStream(assetFd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFd.startOffset,
                assetFd.declaredLength
            )
        }
    }
}
