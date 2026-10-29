package com.developer27.xemotion.videoprocessing.drawing

import com.developer27.xemotion.videoprocessing.AppearanceSettings
import kotlin.math.roundToInt

/** Immutable per-frame appearance: changing one mode never changes an in-flight frame. */
data class OverlayStyle(
    val fontScale: Double,
    val textThickness: Int,
    val padding: Int,
    val margin: Int,
    val stackGap: Int,
    val boxThickness: Int,
    val traceThickness: Int
) {
    companion object {
        val HANDHELD: OverlayStyle get() = capture(false)
        val AR: OverlayStyle get() = capture(true)

        fun capture(arMode: Boolean): OverlayStyle {
            val profile = AppearanceSettings.profile(
                if (arMode) AppearanceSettings.Mode.AR else AppearanceSettings.Mode.CV
            )
            val textRatio = profile.textSize / 16.0
            return OverlayStyle(
                fontScale = profile.textSize / 10.0,
                textThickness = if (arMode) 1 else (3 * textRatio).roundToInt().coerceAtLeast(1),
                padding = (6 * textRatio).roundToInt().coerceAtLeast(1),
                margin = (8 * textRatio).roundToInt().coerceAtLeast(1),
                stackGap = (3 * textRatio).roundToInt().coerceAtLeast(1),
                boxThickness = profile.boxThickness,
                traceThickness = profile.traceThickness
            )
        }
    }
}
