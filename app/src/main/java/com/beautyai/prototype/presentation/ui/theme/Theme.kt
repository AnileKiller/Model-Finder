package com.beautyai.prototype.presentation.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// ── Beauty AI brand palette ──────────────────────────────────────────────────
private val Pink80    = Color(0xFFFFB3C6)
private val PinkGrey  = Color(0xFF7D5260)
private val Pink40    = Color(0xFF9C4068)

private val DarkScheme = darkColorScheme(
    primary         = Pink80,
    secondary       = Color(0xFFEFB8C8),
    tertiary        = Color(0xFFFFB4AB),
    background      = Color(0xFF1A1215),
    surface         = Color(0xFF261820),
    onPrimary       = Color(0xFF561E33),
    onBackground    = Color(0xFFECDFE2),
    onSurface       = Color(0xFFECDFE2)
)

private val LightScheme = lightColorScheme(
    primary         = Pink40,
    secondary       = PinkGrey,
    tertiary        = Color(0xFF7D2D00),
    background      = Color(0xFFFFFBFF),
    surface         = Color(0xFFFFF8F9),
    onPrimary       = Color.White,
    onBackground    = Color(0xFF201115),
    onSurface       = Color(0xFF201115)
)

@Composable
fun BeautyAITheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkScheme
        else      -> LightScheme
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography  = Typography,
        content     = content
    )
}
