package com.beautyai.prototype.domain.usecase

import android.graphics.Bitmap
import com.beautyai.prototype.data.repository.FaceAnalysisRepository
import com.beautyai.prototype.domain.model.FaceData

/**
 * Single-responsibility use-case: run the three-model analysis pipeline
 * and return [FaceData] (or null when no face is found).
 */
class AnalyseFaceUseCase(private val repository: FaceAnalysisRepository) {

    suspend operator fun invoke(
        bitmap: Bitmap,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): FaceData? = repository.analyse(bitmap, onProgress)
}
