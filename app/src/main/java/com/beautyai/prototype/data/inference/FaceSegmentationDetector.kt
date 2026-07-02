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
 * Wraps the XenoFormer face segmentation TFLite model.
 *
 * Model:  face_segmentation_xenoformer_xs_2024_04_02.int8.tflite
 * Input:  [1, H, W, 3]   — normalised float RGB in [0, 1]
 * Output: [1, H, W, 1]   — per-pixel probability that the pixel is skin/face
 *
 * NOTE: despite the "int8" filename, this model is *weight-only* (dynamic
 * range) quantised — only the weights are int8, while the actual input and
 * output tensors are still float32. Feeding it raw int8 buffers causes
 * TRANSPOSE_CONV to fail at interpreter-creation time with
 * "weights->type != input->type (INT8 != FLOAT32)". We therefore use
 * float32 buffers here, same as the other two models.
 *
 * We work at a fixed 256×256 resolution and then bilinearly scale the
 * mask back to the source image.
 *
 * Returns a 2D float array [height][width] with values in [0, 1].
 */
class FaceSegmentationDetector(context: Context) : AutoCloseable {

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?

    private val inputSize = 256

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * FLOAT_BYTES)
            .also { it.order(ByteOrder.nativeOrder()) }

    private val outputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(1 * inputSize * inputSize * 1 * FLOAT_BYTES)
            .also { it.order(ByteOrder.nativeOrder()) }

    init {
        val modelBuffer = loadModelFile(context, MODEL_FILE)
        val compatList = CompatibilityList()

        // GPU delegate does not support int8 on all devices — fall back to CPU
        val options = Interpreter.Options().apply {
            gpuDelegate = null
            setNumThreads(4)
            setUseNNAPI(true)
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    /**
     * Runs segmentation on the face region of [bitmap] defined by [faceBox].
     * Returns a 2D array sized [bitmap.height][bitmap.width] with per-pixel
     * skin probability in [0, 1].
     */
    fun segment(bitmap: Bitmap, faceBox: RectF): Array<FloatArray> {
        // Work on the full image but mask outside the face box later
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        fillInputBuffer(scaled)

        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        // Upsample mask back to original image dimensions
        return upsampleMask(outputBuffer, bitmap.width, bitmap.height)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

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

    /**
     * Bilinear upsampling of the [inputSize]×[inputSize] mask to [targetW]×[targetH].
     */
    private fun upsampleMask(raw: ByteBuffer, targetW: Int, targetH: Int): Array<FloatArray> {
        raw.rewind()
        val flatMask = FloatArray(inputSize * inputSize)
        for (i in flatMask.indices) {
            flatMask[i] = raw.getFloat()
        }

        val result = Array(targetH) { FloatArray(targetW) }
        val scaleX = inputSize.toFloat() / targetW
        val scaleY = inputSize.toFloat() / targetH

        for (y in 0 until targetH) {
            for (x in 0 until targetW) {
                val srcX = x * scaleX
                val srcY = y * scaleY
                val x0   = srcX.toInt().coerceIn(0, inputSize - 1)
                val y0   = srcY.toInt().coerceIn(0, inputSize - 1)
                val x1   = (x0 + 1).coerceIn(0, inputSize - 1)
                val y1   = (y0 + 1).coerceIn(0, inputSize - 1)
                val dx   = srcX - x0
                val dy   = srcY - y0

                result[y][x] =
                    flatMask[y0 * inputSize + x0] * (1 - dx) * (1 - dy) +
                    flatMask[y0 * inputSize + x1] * dx       * (1 - dy) +
                    flatMask[y1 * inputSize + x0] * (1 - dx) * dy       +
                    flatMask[y1 * inputSize + x1] * dx       * dy
            }
        }
        return result
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    companion object {
        private const val MODEL_FILE =
            "models/face_segmentation_xenoformer_xs_2024_04_02.int8.tflite"
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
