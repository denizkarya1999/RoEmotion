package com.developer27.xemotion.videoprocessing.drawing

import java.util.Locale

/** Formats separate YOLO-user and emotion labels for a tracked LED. */
internal object InferenceLabelFormatter {
    data class Labels(
        val user: String,
        val emotion: String?
    )

    fun format(
        userNumber: Int,
        detectionConfidence: Float,
        emotion: String,
        emotionConfidencePct: Float
    ): Labels {
        val userLabel =
            "User ${userNumber.coerceAtLeast(1)} - ${formatPercent(detectionConfidence * 100f)}"
        val normalizedEmotion = emotion.trim()
        val emotionLabel = when {
            normalizedEmotion.isEmpty() -> null
            !emotionConfidencePct.isFinite() -> normalizedEmotion
            else -> "$normalizedEmotion - ${formatPercent(emotionConfidencePct)}"
        }
        return Labels(user = userLabel, emotion = emotionLabel)
    }

    private fun formatPercent(value: Float): String =
        String.format(Locale.US, "%.0f%%", value.coerceIn(0f, 100f))
}
