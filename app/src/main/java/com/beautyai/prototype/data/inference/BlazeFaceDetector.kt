package com.beautyai.prototype.data.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Wraps the BlazeFace short-range TFLite model for face detection.
 *
 * Model:  blaze_face_short_range.tflite
 * Input:  [1, 128, 128, 3]  — normalised float RGB in [-1, 1]
 * Output: Two tensors:
 *           - regressors  [1, 896, 16] bounding-box regressions
 *           - classificators [1, 896, 1] confidence scores
 *
 * The model uses anchor-based detection. This implementation applies
 * a simple NMS pass and returns the single highest-confidence box.
 *
 * Thread safety: NOT thread-safe. Create one instance per thread or
 * guard calls with a mutex.
 */
class BlazeFaceDetector(context: Context) : AutoCloseable {

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?

    private val inputSize = 128
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * FLOAT_BYTES)
            .also { it.order(ByteOrder.nativeOrder()) }

    // Output tensors — allocated once and reused.
    // Shape: [1, 896, 16]  (bounding box regressions)
    private val regressors = Array(1) { Array(896) { FloatArray(16) } }

    // Shape: [1, 896, 1]  (class scores)
    private val scores = Array(1) { Array(896) { FloatArray(1) } }

    // Pre-computed anchors matching the short-range BlazeFace spec
    private val anchors: Array<FloatArray> = generateAnchors()

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
     * Runs BlazeFace inference on [bitmap] and returns the bounding box
     * of the most confident face detection in normalised [0,1] coordinates,
     * or null if no face is found above [confidenceThreshold].
     */
    fun detect(bitmap: Bitmap, confidenceThreshold: Float = 0.7f): RectF? {
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        fillInputBuffer(scaled)

        val outputMap = mapOf(
            0 to regressors,
            1 to scores
        )
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        return decodeBoxes(
            regressors[0], scores[0],
            imageWidth = bitmap.width.toFloat(),
            imageHeight = bitmap.height.toFloat(),
            threshold = confidenceThreshold
        )
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in pixels) {
            // Normalise each channel to [-1, 1]
            inputBuffer.putFloat(((pixel shr 16 and 0xFF) / 127.5f) - 1f) // R
            inputBuffer.putFloat(((pixel shr 8  and 0xFF) / 127.5f) - 1f) // G
            inputBuffer.putFloat(((pixel        and 0xFF) / 127.5f) - 1f) // B
        }
    }

    private fun decodeBoxes(
        regressions: Array<FloatArray>,
        classScores: Array<FloatArray>,
        imageWidth: Float,
        imageHeight: Float,
        threshold: Float
    ): RectF? {
        var bestScore = threshold
        var bestBox: RectF? = null

        for (i in regressions.indices) {
            val score = sigmoid(classScores[i][0])
            if (score <= bestScore) continue

            val anchor = anchors[i]
            val cx = regressions[i][0] / inputSize + anchor[0]
            val cy = regressions[i][1] / inputSize + anchor[1]
            val w  = regressions[i][2] / inputSize
            val h  = regressions[i][3] / inputSize

            bestScore = score
            bestBox = RectF(
                ((cx - w / 2f) * imageWidth).coerceIn(0f, imageWidth),
                ((cy - h / 2f) * imageHeight).coerceIn(0f, imageHeight),
                ((cx + w / 2f) * imageWidth).coerceIn(0f, imageWidth),
                ((cy + h / 2f) * imageHeight).coerceIn(0f, imageHeight)
            )
        }
        return bestBox
    }

    private fun sigmoid(x: Float) = 1f / (1f + Math.exp(-x.toDouble()).toFloat())

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    // ── Static helpers ────────────────────────────────────────────────────────

    companion object {
        private const val MODEL_FILE = "models/blaze_face_short_range.tflite"
        private const val FLOAT_BYTES = 4

        private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
            val assetFd = context.assets.openFd(filename)
            return FileInputStream(assetFd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFd.startOffset,
                assetFd.declaredLength
            )
        }

        /**
         * Generates the 896 anchor boxes for the short-range BlazeFace model.
         * Strides and scales follow the original MediaPipe configuration.
         */
        private fun generateAnchors(): Array<FloatArray> {
            val strides = intArrayOf(8, 16, 16, 16)
            val anchorsPerStride = intArrayOf(2, 6, 6, 6)
            val inputSize = 128f
            val anchors = mutableListOf<FloatArray>()

            for ((strideIdx, stride) in strides.withIndex()) {
                val gridRows = kotlin.math.ceil(inputSize / stride).toInt()
                val gridCols = kotlin.math.ceil(inputSize / stride).toInt()
                val anchorsCount = anchorsPerStride[strideIdx]

                for (y in 0 until gridRows) {
                    for (x in 0 until gridCols) {
                        repeat(anchorsCount) {
                            anchors.add(floatArrayOf(
                                (x + 0.5f) / gridCols,
                                (y + 0.5f) / gridRows
                            ))
                        }
                    }
                }
            }
            return anchors.toTypedArray()
        }
    }
}
