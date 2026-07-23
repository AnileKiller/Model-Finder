package com.beautyai.prototype.presentation.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.beautyai.prototype.domain.model.BeautyParameters

/**
 * Renders all nine beauty sliders in a vertical list.
 * Each slider is fully self-contained and calls [onParamsChange] with an
 * updated copy of [params] when moved.
 */
@Composable
fun BeautySliders(
    params: BeautyParameters,
    onParamsChange: (BeautyParameters) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        BeautySliderRow(
            label = "Skin Brightness",
            value = params.skinBrightness,
            onValueChange = { onParamsChange(params.copy(skinBrightness = it)) }
        )
        BeautySliderRow(
            label = "Skin Tone",
            value = params.skinToneEnhancement,
            onValueChange = { onParamsChange(params.copy(skinToneEnhancement = it)) }
        )
        BeautySliderRow(
            label = "Blemish Reduction",
            value = params.blemishReduction,
            onValueChange = { onParamsChange(params.copy(blemishReduction = it)) }
        )
        BeautySliderRow(
            label = "Under-Eye",
            value = params.underEyeReduction,
            onValueChange = { onParamsChange(params.copy(underEyeReduction = it)) }
        )
        BeautySliderRow(
            label = "Eye Brightness",
            value = params.eyeBrightness,
            onValueChange = { onParamsChange(params.copy(eyeBrightness = it)) }
        )
        BeautySliderRow(
            label = "Teeth Whitening",
            value = params.teethWhitening,
            onValueChange = { onParamsChange(params.copy(teethWhitening = it)) }
        )
        BeautySliderRow(
            label = "Face Sharpening",
            value = params.faceSharpening,
            onValueChange = { onParamsChange(params.copy(faceSharpening = it)) }
        )
        BeautySliderRow(
            label = "Eyebrow Definition",
            value = params.eyebrowDefinition,
            onValueChange = { onParamsChange(params.copy(eyebrowDefinition = it)) }
        )

        Spacer(Modifier.height(8.dp))
        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
        Spacer(Modifier.height(4.dp))

        BeautySliderRow(
            label = "Overall Intensity",
            value = params.overallIntensity,
            onValueChange = { onParamsChange(params.copy(overallIntensity = it)) },
            isPrimary = true
        )
    }
}

@Composable
private fun BeautySliderRow(
    label: String,
    value: Float,
    onValueChange: (Float) -> Unit,
    isPrimary: Boolean = false
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium.copy(
                    fontWeight = if (isPrimary) FontWeight.SemiBold else FontWeight.Normal,
                    fontSize   = if (isPrimary) 14.sp else 13.sp
                ),
                color = if (isPrimary)
                    MaterialTheme.colorScheme.primary
                else
                    MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "${(value * 100).toInt()}%",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = 0f..1f,
            modifier = Modifier
                .fillMaxWidth()
                .height(32.dp),
            colors = if (isPrimary) SliderDefaults.colors(
                thumbColor       = MaterialTheme.colorScheme.primary,
                activeTrackColor = MaterialTheme.colorScheme.primary
            ) else SliderDefaults.colors()
        )
    }
}
