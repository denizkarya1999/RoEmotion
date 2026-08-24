package com.developer27.xemotion.videoprocessing.drawing

import org.junit.Assert.assertEquals
import org.junit.Test

class InferenceLabelFormatterTest {
    @Test
    fun labelsSeparateUserAboveAndEmotionBelow() {
        assertEquals(
            InferenceLabelFormatter.Labels(
                user = "User 1 - 100%",
                emotion = "Disgust - 99%"
            ),
            InferenceLabelFormatter.format(
                userNumber = 1,
                detectionConfidence = 0.999f,
                emotion = "Disgust",
                emotionConfidencePct = 99.1f
            )
        )
    }

    @Test
    fun detectionOnlyLabelStillUsesCompactFormat() {
        assertEquals(
            InferenceLabelFormatter.Labels(
                user = "User 2 - 46%",
                emotion = null
            ),
            InferenceLabelFormatter.format(
                userNumber = 2,
                detectionConfidence = 0.456f,
                emotion = "",
                emotionConfidencePct = Float.NaN
            )
        )
    }
}
