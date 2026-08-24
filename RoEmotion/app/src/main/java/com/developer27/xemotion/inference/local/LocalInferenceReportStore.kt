package com.developer27.xemotion.inference.local

import android.content.Context
import android.net.Uri
import com.developer27.xemotion.storage.MediaStoreRepository
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class LocalInferenceReportStore(context: Context) {
    private val storage = MediaStoreRepository(context)

    fun save(fileName: String, report: String): Uri =
        storage.writeDocument(fileName, report, REPORT_DIRECTORY)

    fun buildSummary(fileName: String, uri: Uri?, report: String): String {
        val summaryLines = report.lines().filter { line ->
            line.startsWith("Overall ") ||
                line.contains("Percentage:") ||
                line.startsWith("Success:") ||
                line.contains("Image-Level Full Match")
        }
        return buildString {
            appendLine("Log file saved.")
            appendLine()
            appendLine("File Name: $fileName")
            uri?.let { appendLine("Saved URI: $it") }
            appendLine()
            appendLine("Summary:")
            summaryLines.take(MAX_SUMMARY_LINES).forEach(::appendLine)
        }.trim()
    }

    fun timestamp(): String = SimpleDateFormat(DATE_PATTERN, Locale.US).format(Date())
    fun fileTimestamp(): String = SimpleDateFormat(FILE_PATTERN, Locale.US).format(Date())

    private companion object {
        const val REPORT_DIRECTORY = "RoEmotion"
        const val MAX_SUMMARY_LINES = 20
        const val DATE_PATTERN = "yyyy-MM-dd HH:mm:ss"
        const val FILE_PATTERN = "yyyyMMdd_HHmmss"
    }
}
