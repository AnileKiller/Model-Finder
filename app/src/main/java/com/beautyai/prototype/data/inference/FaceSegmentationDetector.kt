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
 * Wraps Google's publicly released MediaPipe "Selfie Multiclass" segmentation
 * model (part of the MediaPipe Image Segmenter task).
 *
 * Model:  selfie_multiclass_256x256.tflite
 *   https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
 *
 * Input:  [1, 256, 256, 3] — float32 RGB normalised to [0, 1]
 * Output: [1, 256, 256, 6] — float32 per-pixel per-class confidence, classes:
 *   0 = background, 1 = hair, 2 = body-skin, 3 = face-skin, 4 = clothes, 5 = other
 *
 * This replaces the previously used "XenoFormer" model, which was extracted
 * from the Snapseed app and targeted an unreleased/internal TFLite runtime
 * (2.21.0). That model's hybrid INT8/FLOAT32 TRANSPOSE_CONV op could not be
 * validated by any publicly available TFLite SDK version and crashed at
 * interpreter creation regardless of buffer type, XNNPACK setting, or runtime
 * version. This model is fully public, float32 throughout, and designed to
 * run on the standard TFLite Android runtime.
 *
 * We extract the "face-skin" channel (index 3) as the beauty-retouch mask,
 * work at a fixed 256x256 resolution, then bilinearly scale it back up to
 * the source image size.
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
        ByteBuffer.allocateDirect(1 * inputSize * inputSize * NUM_CLASSES * FLOAT_BYTES)
            .also { it.order(ByteOrder.nativeOrder()) }

    init {
        val modelBuffer = loadModelFile(context, MODEL_FILE)
        val compatList = CompatibilityList()

        val options = Interpreter.Options().apply {
            gpuDelegate = null
            setNumThreads(4)
            setUseNNAPI(true)
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    /**
     * Runs segmentation on [bitmap]. [faceBox] is currently unused (the model
     * segments the whole frame) but kept for API compatibility with callers
     * and potential future face-region cropping.
     *
     * Returns a 2D array sized [bitmap.height][bitmap.width] with per-pixel
     * face-skin probability in [0, 1].
     */
    fun segment(bitmap: Bitmap, faceBox: RectF): Array<FloatArray> {
        val scaled = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        fillInputBuffer(scaled)

        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

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
     * Extracts the face-skin channel ([FACE_SKIN_CLASS_INDEX]) from the
     * multi-class output and bilinearly upsamples it to [targetW]×[targetH].
     */
    private fun upsampleMask(raw: ByteBuffer, targetW: Int, targetH: Int): Array<FloatArray> {
        raw.rewind()
        val flatMask = FloatArray(inputSize * inputSize)
        for (i in 0 until inputSize * inputSize) {
            var faceSkinScore = 0f
            for (c in 0 until NUM_CLASSES) {
                val value = raw.getFloat()
                if (c == FACE_SKIN_CLASS_INDEX) faceSkinScore = value
            }
            flatMask[i] = faceSkinScore
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
        private const val MODEL_FILE = "models/selfie_multiclass_256x256.tflite"
        private const val FLOAT_BYTES = 4
        private const val NUM_CLASSES = 6

        // MediaPipe selfie_multiclass_256x256 class indices:
        // 0 = background, 1 = hair, 2 = body-skin, 3 = face-skin, 4 = clothes, 5 = other
        private const val FACE_SKIN_CLASS_INDEX = 3

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
