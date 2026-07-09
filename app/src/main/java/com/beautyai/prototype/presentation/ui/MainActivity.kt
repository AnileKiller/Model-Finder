package com.beautyai.prototype.presentation.ui

import android.Manifest
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.beautyai.prototype.presentation.ui.theme.BeautyAITheme
import com.beautyai.prototype.presentation.viewmodel.BeautyViewModel

/**
 * Single-activity host. Handles runtime permission requests before delegating
 * all UI work to [MainScreen] (Jetpack Compose).
 */
class MainActivity : ComponentActivity() {

    private val viewModel: BeautyViewModel by viewModels()

    // ── Permission launcher ──────────────────────────────────────────────────

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* permissions granted or denied — gallery picker handles its own flow */ }

    // ── Image picker ─────────────────────────────────────────────────────────

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { viewModel.loadImage(it) }
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestStoragePermissions()

        setContent {
            BeautyAITheme {
                val processingState  by viewModel.processingState.collectAsState()
                val beautyParams     by viewModel.beautyParams.collectAsState()
                val isReprocessing   by viewModel.isReprocessing.collectAsState()
                val showOriginal     by viewModel.showOriginal.collectAsState()
                val showMaskOverlay  by viewModel.showMaskOverlay.collectAsState()
                val saveSuccess      by viewModel.saveSuccess.collectAsState()
                val blemishDebugLog  by viewModel.blemishDebugLog.collectAsState()

                MainScreen(
                    processingState     = processingState,
                    beautyParams        = beautyParams,
                    isReprocessing      = isReprocessing,
                    showOriginal        = showOriginal,
                    showMaskOverlay     = showMaskOverlay,
                    saveSuccess         = saveSuccess,
                    blemishDebugLog     = blemishDebugLog,
                    onSelectImage       = { imagePickerLauncher.launch("image/*") },
                    onParamsChange      = viewModel::updateParams,
                    onReset             = viewModel::resetParams,
                    onToggleOriginal    = viewModel::toggleOriginal,
                    onToggleMaskOverlay = viewModel::toggleMaskOverlay,
                    onSave              = viewModel::saveImage,
                    onDismissSave       = viewModel::clearSaveMessage
                )
            }
        }
    }

    private fun requestStoragePermissions() {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
        }
        permissionLauncher.launch(permissions)
    }
}
