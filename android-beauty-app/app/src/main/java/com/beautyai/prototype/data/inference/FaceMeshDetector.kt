package com.beautyai.prototype.data.inference

import android.content.Context
import android.graphics.Bitmap
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
 * Wraps the FaceMesh lite TFLite model to extract 468 facial landmarks.
 *
 * Model:  facemesh_lite_468_2022_09_06.f16.tflite
 * Input:  [1, 192, 192, 3] — normalised float RGB in [0, 1]
 * Output:
 *   - landmarks  [1, 1, 1, 1404]  (468 × 3 values: x, y, z in [0, 192])
 *   - score      [1, 1]           (face presence confidence)
 *
 * Landmarks are returned in normalised [0,1] image coordinates.
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
    private val scoreOut     = Array(1) { FloatArray(1) }

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
     */
    fun detect(bitmap: Bitmap, faceBox: RectF, presenceThreshold: Float = 0.5f): List<Landmark>? {
        // Crop the detected face region and scale to 192×192
        val cropped = cropAndScale(bitmap, faceBox)
        fillInputBuffer(cropped)

        val outputMap = mapOf(
            0 to landmarksOut,
            1 to scoreOut
        )
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        val presenceScore = sigmoid(scoreOut[0][0])
        if (presenceScore < presenceThreshold) return null

        // Denormalise from [0, 192] crop-space back to original image space
        val cropX = faceBox.left / bitmap.width
        val cropY = faceBox.top  / bitmap.height
        val cropW = faceBox.width()  / bitmap.width
        val cropH = faceBox.height() / bitmap.height

        val flat = landmarksOut[0][0][0]
        return List(468) { i ->
            val normX = flat[i * 3]     / inputSize
            val normY = flat[i * 3 + 1] / inputSize
            val z     = flat[i * 3 + 2]
            Landmark(
                x = cropX + normX * cropW,
                y = cropY + normY * cropH,
                z = z
            )
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private fun cropAndScale(src: Bitmap, box: RectF): Bitmap {
        val x = box.left.toInt().coerceAtLeast(0)
        val y = box.top.toInt().coerceAtLeast(0)
        val w = box.width().toInt().coerceAtMost(src.width - x)
        val h = box.height().toInt().coerceAtMost(src.height - y)
        val cropped = Bitmap.createBitmap(src, x, y, w, h)
        return Bitmap.createScaledBitmap(cropped, inputSize, inputSize, true)
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
        private const val MODEL_FILE = "models/facemesh_lite_468_2022_09_06.f16.tflite"
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
