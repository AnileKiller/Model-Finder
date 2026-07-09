package com.beautyai.prototype.presentation.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.beautyai.prototype.data.repository.FaceAnalysisRepository
import com.beautyai.prototype.data.repository.ImageRepository
import com.beautyai.prototype.domain.model.BeautyParameters
import com.beautyai.prototype.domain.model.FaceData
import com.beautyai.prototype.domain.model.ProcessingState
import com.beautyai.prototype.domain.usecase.AnalyseFaceUseCase
import com.beautyai.prototype.domain.usecase.ApplyBeautyUseCase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * MVVM ViewModel for the beauty-enhancement screen.
 *
 * Responsibilities:
 *  - Expose [processingState] and [beautyParams] as [StateFlow]s to the UI.
 *  - Orchestrate the three-stage pipeline (face detection → enhancement).
 *  - Debounce slider updates so the pipeline doesn't saturate the CPU.
 *  - Delegate all heavy work to [Dispatchers.Default].
 */
class BeautyViewModel(application: Application) : AndroidViewModel(application) {

    // ── Repositories & use-cases ─────────────────────────────────────────────

    private val imageRepo      = ImageRepository(application)
    private val faceRepo       = FaceAnalysisRepository(application)
    private val analyseFace    = AnalyseFaceUseCase(faceRepo)
    private val applyBeauty    = ApplyBeautyUseCase()

    // ── UI state ─────────────────────────────────────────────────────────────

    private val _processingState = MutableStateFlow<ProcessingState>(ProcessingState.Idle)
    val processingState: StateFlow<ProcessingState> = _processingState.asStateFlow()

    private val _beautyParams = MutableStateFlow(BeautyParameters.DEFAULT)
    val beautyParams: StateFlow<BeautyParameters> = _beautyParams.asStateFlow()

    /** True while an enhanced image is being applied after a slider change. */
    private val _isReprocessing = MutableStateFlow(false)
    val isReprocessing: StateFlow<Boolean> = _isReprocessing.asStateFlow()

    private val _showOriginal = MutableStateFlow(false)
    val showOriginal: StateFlow<Boolean> = _showOriginal.asStateFlow()

    private val _saveSuccess = MutableStateFlow<String?>(null)
    val saveSuccess: StateFlow<String?> = _saveSuccess.asStateFlow()

    private val _showMaskOverlay = MutableStateFlow(false)
    val showMaskOverlay: StateFlow<Boolean> = _showMaskOverlay.asStateFlow()

    private val _blemishDebugLog = MutableStateFlow("")
    val blemishDebugLog: StateFlow<String> = _blemishDebugLog.asStateFlow()

    // ── Cached analysis results ──────────────────────────────────────────────

    /** Holds the last successfully analysed face so slider tweaks don't re-run inference. */
    private var cachedFaceData: FaceData? = null

    /** Holds the last clean (non-overlaid) enhanced bitmap for overlay toggling. */
    private var cachedEnhanced: android.graphics.Bitmap? = null

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Load an image from [uri], run the full AI pipeline, and publish the result.
     */
    fun loadImage(uri: Uri) {
        viewModelScope.launch {
            _processingState.value = ProcessingState.Processing(0f, "Loading image…")
            cachedFaceData = null

            runCatching {
                val bitmap = imageRepo.loadBitmap(uri)

                _processingState.value = ProcessingState.Processing(0.05f, "Analysing face…")

                val faceData = analyseFace(bitmap) { progress, msg ->
                    _processingState.value = ProcessingState.Processing(
                        progress = 0.05f + progress * 0.7f,
                        message  = msg
                    )
                }

                _processingState.value = ProcessingState.Processing(0.85f, "Applying beauty…")
                cachedFaceData = faceData

                val enhanced = withContext(Dispatchers.Default) {
                    if (faceData != null) {
                        val logLines = mutableListOf<String>()
                        val result = applyBeauty(bitmap, faceData, _beautyParams.value) { msg ->
                            if (logLines.size < 5) logLines.add(msg)
                        }
                        if (logLines.isNotEmpty()) _blemishDebugLog.value = logLines.joinToString("\n")
                        result
                    } else {
                        bitmap
                    }
                }
                cachedEnhanced = enhanced

                val displayed = withContext(Dispatchers.Default) {
                    if (_showMaskOverlay.value && faceData != null) {
                        applyBeauty.renderMaskDebugOverlay(enhanced, faceData)
                    } else {
                        enhanced
                    }
                }

                _processingState.value = ProcessingState.Success(
                    original = bitmap,
                    enhanced = displayed,
                    faceData = faceData
                )
            }.onFailure { e ->
                _processingState.value = ProcessingState.Error(
                    e.localizedMessage ?: "Unknown error"
                )
            }
        }
    }

    /**
     * Called whenever a slider is moved. Re-applies the beauty pipeline to the
     * cached original bitmap without re-running face inference.
     */
    fun updateParams(params: BeautyParameters) {
        _beautyParams.value = params
        val current = _processingState.value
        if (current !is ProcessingState.Success) return

        viewModelScope.launch {
            _isReprocessing.value = true
            runCatching {
                val enhanced = withContext(Dispatchers.Default) {
                    val faceData = cachedFaceData
                    if (faceData != null) {
                        val logLines = mutableListOf<String>()
                        val result = applyBeauty(current.original, faceData, params) { msg ->
                            if (logLines.size < 5) logLines.add(msg)
                        }
                        if (logLines.isNotEmpty()) _blemishDebugLog.value = logLines.joinToString("\n")
                        result
                    } else {
                        current.original
                    }
                }
                cachedEnhanced = enhanced

                val displayed = withContext(Dispatchers.Default) {
                    val faceData = cachedFaceData
                    if (_showMaskOverlay.value && faceData != null) {
                        applyBeauty.renderMaskDebugOverlay(enhanced, faceData)
                    } else {
                        enhanced
                    }
                }
                _processingState.value = current.copy(enhanced = displayed)
            }
            _isReprocessing.value = false
        }
    }

    /** Reset all sliders to their default values and re-apply. */
    fun resetParams() {
        updateParams(BeautyParameters.DEFAULT)
    }

    /** Toggle between showing the original and the enhanced image. */
    fun toggleOriginal() {
        _showOriginal.update { !it }
    }

    /**
     * Toggle the debug mask overlay on/off.
     * Green = active skin zone, Red = excluded feature zone (eyes/lips/etc).
     */
    fun toggleMaskOverlay() {
        val current = _processingState.value as? ProcessingState.Success ?: run {
            _showMaskOverlay.update { !it }
            return
        }
        val faceData = cachedFaceData
        val base = cachedEnhanced ?: current.enhanced
        _showMaskOverlay.update { !it }
        val nowOn = _showMaskOverlay.value

        viewModelScope.launch {
            _isReprocessing.value = true
            runCatching {
                val displayed = withContext(Dispatchers.Default) {
                    if (nowOn && faceData != null) {
                        applyBeauty.renderMaskDebugOverlay(base, faceData)
                    } else {
                        base
                    }
                }
                _processingState.value = current.copy(enhanced = displayed)
            }
            _isReprocessing.value = false
        }
    }

    /** Save the currently enhanced image to the gallery. */
    fun saveImage() {
        val current = _processingState.value as? ProcessingState.Success ?: return
        viewModelScope.launch {
            runCatching {
                val uri = imageRepo.saveBitmap(current.enhanced)
                _saveSuccess.value = "Image saved to gallery"
            }.onFailure { e ->
                _saveSuccess.value = "Save failed: ${e.localizedMessage}"
            }
        }
    }

    fun clearSaveMessage() {
        _saveSuccess.value = null
    }

    override fun onCleared() {
        super.onCleared()
        faceRepo.close()
    }
}
