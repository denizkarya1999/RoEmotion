package com.developer27.xemotion.storage

import org.junit.Assert.assertEquals
import org.junit.Test
import java.text.SimpleDateFormat
import java.util.Locale

class OokVideoNamingTest {
    @Test
    fun videoFileNameContainsUserModulationAndTimestampInRequestedOrder() {
        val capturedAt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
            .parse("2026-08-31 15:42:07.219")!!

        assertEquals(
            "User 01_1000 Hz OOK_2026_08_31_15_42_07_219.mp4",
            ookVideoFileName("User 01", "1000 Hz OOK", capturedAt)
        )
    }

    @Test
    fun videoFileNameSanitizesEachUserProvidedSegment() {
        val capturedAt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
            .parse("2026-08-31 15:42:07.219")!!

        assertEquals(
            "User _ 01_PWM_OOK_2026_08_31_15_42_07_219.mp4",
            ookVideoFileName("User / 01", "PWM:OOK", capturedAt)
        )
    }
}
