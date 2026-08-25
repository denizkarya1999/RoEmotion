package com.developer27.xemotion.videoprocessing

import android.graphics.Bitmap
import android.util.Log
import com.developer27.xemotion.storage.MediaStoreRepository
import com.developer27.xemotion.videoprocessing.drawing.traces.TracePreprocessing
import org.opencv.core.Point
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

internal const val COLLECTED_DATA_ROOT = "Collected RoEmotion Data"

internal fun sanitizeCollectionPathSegment(value: String): String = value
    .trim()
    .replace(Regex("""[\\/:*?\"<>|\p{Cntrl}]"""), "_")
    .trim(' ', '.')
    .take(80)
    .trim(' ', '.')
    .ifBlank { "Unspecified" }

internal fun collectionDirectory(
    userName: String,
    emotionType: String,
    traceType: Settings.Trace.Type
): String = "$COLLECTED_DATA_ROOT/${sanitizeCollectionPathSegment(userName)}/" +
    "${sanitizeCollectionPathSegment(emotionType)}/${traceType.displayName}"

internal fun traceFileName(
    capturedAt: Date,
    traceType: Settings.Trace.Type,
    userName: String
): String {
    val timestamp = SimpleDateFormat("yyyy_MM_dd_HH_mm_ss_SSS", Locale.US).format(capturedAt)
    return "${timestamp}_${traceType.name}_${sanitizeCollectionPathSegment(userName)}.jpg"
}

/** Generates and persists trace-line images. */
class LineProcessing(
    private val storage: MediaStoreRepository
) {
    private val tracePreprocessing = TracePreprocessing()

    var batchCount = 0
        private set
    var savedTraceCount = 0
        private set
    private var collectionUserName = "Unspecified"
    private var collectionEmotion = "Unspecified"

    fun beginSession(userName: String? = null, emotionType: String? = null) {
        batchCount = 0
        savedTraceCount = 0
        collectionUserName = sanitizeCollectionPathSegment(userName.orEmpty())
        collectionEmotion = sanitizeCollectionPathSegment(emotionType.orEmpty())
    }

    fun exportProcessedTraceForClass(
        points: List<Point>,
        type: Settings.Trace.Type
    ) = tracePreprocessing.exportProcessedTraceForClass(points, type)

    fun exportSelectedContourTraces(
        rawPoints: List<Point>,
        smoothPoints: List<Point>,
        types: Set<Settings.Trace.Type>
    ): Map<Settings.Trace.Type, Bitmap> = buildMap {
        types.sortedBy { it.ordinal }.forEach { type ->
            val points = when (type) {
                Settings.Trace.Type.RAW,
                Settings.Trace.Type.RAW_CV -> rawPoints
                Settings.Trace.Type.SPLINE,
                Settings.Trace.Type.SPLINE_CV -> smoothPoints
            }
            tracePreprocessing.exportTrace(points, type)?.let { put(type, it) }
        }
    }

    fun saveContourTraces(traces: Map<Settings.Trace.Type, Bitmap>) {
        if (traces.isEmpty()) return
        val capturedAt = Date()
        var savedAny = false
        traces.forEach { (type, bitmap) ->
            val directory = collectionDirectory(collectionUserName, collectionEmotion, type)
            val fileName = traceFileName(capturedAt, type, collectionUserName)
            if (storage.saveJpeg(bitmap, directory, fileName)) {
                savedAny = true
                savedTraceCount++
                Log.d(TAG, "Saved $directory/$fileName")
            }
        }
        if (savedAny) {
            batchCount++
        }
    }

    private companion object {
        const val TAG = "LineProcessing"
    }
}
