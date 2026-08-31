package com.developer27.xemotion.storage

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

internal fun ookVideoFileName(
    userName: String,
    modulation: String,
    capturedAt: Date = Date()
): String {
    val timestamp = SimpleDateFormat("yyyy_MM_dd_HH_mm_ss_SSS", Locale.US).format(capturedAt)
    return "${sanitizeVideoFileSegment(userName)}_" +
        "${sanitizeVideoFileSegment(modulation)}_${timestamp}.mp4"
}

internal fun sanitizeVideoFileSegment(value: String): String = value
    .trim()
    .replace(Regex("[\\\\/:*?\"<>|\\p{Cntrl}]"), "_")
    .trim(' ', '.')
    .ifBlank { "Unspecified" }
