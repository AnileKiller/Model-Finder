package com.beautyai.prototype.data.repository

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.OutputStream

/**
 * Handles reading images from the gallery and writing enhanced images back.
 */
class ImageRepository(private val context: Context) {

    /**
     * Loads a [Bitmap] from [uri] at the original resolution.
     * Runs on [Dispatchers.IO].
     */
    suspend fun loadBitmap(uri: Uri): Bitmap = withContext(Dispatchers.IO) {
        context.contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream)
                ?: error("Could not decode image at $uri")
        } ?: error("Could not open input stream for $uri")
    }

    /**
     * Saves [bitmap] to the device's Pictures gallery, preserving full resolution.
     * Returns the [Uri] of the saved file.
     */
    suspend fun saveBitmap(bitmap: Bitmap): Uri = withContext(Dispatchers.IO) {
        val filename = "BeautyAI_${System.currentTimeMillis()}.jpg"

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH,
                "${Environment.DIRECTORY_PICTURES}/BeautyAI")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("MediaStore insert returned null")

        resolver.openOutputStream(uri)?.use { out: OutputStream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
        }

        uri
    }
}
