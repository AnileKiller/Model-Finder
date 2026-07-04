package com.beautyai.prototype.data.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
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
 * We combine the "body-skin" (index 2) and "face-skin" (index 3) channels
 * into a single skin mask — using the max of the two per-pixel confidences —
 * so that the neck and any other visible body skin receive the exact same
 * bilateral-blur beauty treatment as the face, instead of only the face
 * proper. We work at a fixed 256x256 resolution, then bilinearly scale the
 * combined mask back up to the source image size.
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
        val compatList  = CompatibilityList()
        var gpu: GpuDelegate? = null

        // Three-level acceleration fallback:
        //  1. GPU delegate  — hardware-accelerated; fastest on most devices.
        //  2. NNAPI         — Android neural-network API; accelerated on DSP/NPU
        //                     when GPU is unavailable or its delegate throws.
        //  3. CPU (4 threads) — always available; guaranteed not to throw.
        //
        // Each tier is wrapped in try/catch because CompatibilityList is a
        // pre-flight check only — GpuDelegate or Interpreter construction can
        // still throw at runtime on some chipsets/drivers, and NNAPI support
        // varies by device and model op-set. Without the catch the app crashes
        // silently instead of degrading gracefully.
        interpreter = if (compatList.isDelegateSupportedOnThisDevice) {
            try {
                gpu = GpuDelegate(compatList.bestOptionsForThisDevice)
                Interpreter(modelBuffer, Interpreter.Options().addDelegate(gpu))
            } catch (e: Exception) {
                gpu?.close(); gpu = null
                try {
                    Interpreter(modelBuffer, Interpreter.Options()
                        .setUseNNAPI(true).setNumThreads(4))
                } catch (e2: Exception) {
                    Interpreter(modelBuffer, Interpreter.Options().setNumThreads(4))
                }
            }
        } else {
            try {
                Interpreter(modelBuffer, Interpreter.Options()
                    .setUseNNAPI(true).setNumThreads(4))
            } catch (e: Exception) {
                Interpreter(modelBuffer, Interpreter.Options().setNumThreads(4))
            }
        }

        gpuDelegate = gpu
    }

    /**
     * Runs segmentation on [bitmap] and returns a 2D array sized
     * [bitmap.height][bitmap.width] with per-pixel face-skin probability in [0,1].
     *
     * The full image is letterboxed (aspect-ratio-preserving black padding) to
     * 256×256 before inference. The inverse letterbox transform is applied during
     * upsampling so each output pixel maps back to the correct position in the
     * original image. Without this, squishing a non-square image to 256×256 shifts
     * the mask boundary in a direction that depends on the photo's aspect ratio,
     * causing it to misalign with the landmarks produced by FaceMesh.
     */
    fun segment(bitmap: Bitmap, faceBox: RectF): Array<FloatArray> {
        val imgW  = bitmap.width.toFloat()
        val imgH  = bitmap.height.toFloat()

        // Uniform scale that fits the full image inside inputSize×inputSize
        val scale  = minOf(inputSize / imgW, inputSize / imgH)
        val scaledW = (imgW * scale).toInt().coerceAtLeast(1)
        val scaledH = (imgH * scale).toInt().coerceAtLeast(1)
        val padX   = (inputSize - scaledW) / 2f
        val padY   = (inputSize - scaledH) / 2f

        fillInputBuffer(letterboxBitmap(bitmap, scaledW, scaledH))

        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        return upsampleMask(outputBuffer, bitmap.width, bitmap.height, padX, padY, scale)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Scales [src] to [scaledW]×[scaledH] (aspect-ratio-preserving) and centres
     * it on a black [inputSize]×[inputSize] canvas — matching the letterbox
     * approach used in BlazeFaceDetector and FaceMeshDetector.
     */
    private fun letterboxBitmap(src: Bitmap, scaledW: Int, scaledH: Int): Bitmap {
        val scaled      = Bitmap.createScaledBitmap(src, scaledW, scaledH, true)
        val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas      = Canvas(letterboxed)
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

    /**
     * Extracts and combines the body-skin ([BODY_SKIN_CLASS_INDEX]) and
     * face-skin ([FACE_SKIN_CLASS_INDEX]) channels from the raw multi-class
     * output — taking the per-pixel max of the two, since a pixel is skin if
     * either class is confident — then bilinearly resamples the combined
     * mask to [targetW]×[targetH].
     *
     * Each target pixel (x, y) is mapped to its corresponding position inside
     * the 256×256 letterboxed model output using the same [scale], [padX], and
     * [padY] that were used when building the letterboxed input:
     *
     *   srcX = x × scale + padX
     *   srcY = y × scale + padY
     *
     * Pixels that land inside the padding bars (outside the actual image region)
     * produce a srcX/srcY outside the content area; bilinear sampling clamps them
     * to the nearest valid content pixel, which is always a near-zero background
     * value — correctly yielding a 0.0 mask value at the image borders.
     */
    private fun upsampleMask(
        raw: ByteBuffer,
        targetW: Int,
        targetH: Int,
        padX: Float,
        padY: Float,
        scale: Float
    ): Array<FloatArray> {
        raw.rewind()
        val flatMask = FloatArray(inputSize * inputSize)
        for (i in 0 until inputSize * inputSize) {
            var bodySkinScore = 0f
            var faceSkinScore = 0f
            for (c in 0 until NUM_CLASSES) {
                val value = raw.getFloat()
                if (c == BODY_SKIN_CLASS_INDEX) bodySkinScore = value
                if (c == FACE_SKIN_CLASS_INDEX) faceSkinScore = value
            }
            // Combine body-skin (neck, shoulders, etc.) and face-skin into a
            // single skin mask so the neck gets the same treatment as the face.
            flatMask[i] = maxOf(bodySkinScore, faceSkinScore)
        }

        val result = Array(targetH) { FloatArray(targetW) }

        for (y in 0 until targetH) {
            for (x in 0 until targetW) {
                // Map original-image pixel → position in the letterboxed 256×256 output
                val srcX = x * scale + padX
                val srcY = y * scale + padY

                val x0 = srcX.toInt().coerceIn(0, inputSize - 1)
                val y0 = srcY.toInt().coerceIn(0, inputSize - 1)
                val x1 = (x0 + 1).coerceIn(0, inputSize - 1)
                val y1 = (y0 + 1).coerceIn(0, inputSize - 1)
                val dx = srcX - x0
                val dy = srcY - y0

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
        private const val BODY_SKIN_CLASS_INDEX = 2
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
