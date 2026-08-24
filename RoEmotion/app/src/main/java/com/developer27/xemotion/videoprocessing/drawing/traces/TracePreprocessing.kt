package com.developer27.xemotion.videoprocessing.drawing.traces

import android.graphics.Bitmap
import android.graphics.Color
import com.developer27.xemotion.videoprocessing.Settings
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import java.util.Random
import kotlin.math.max
import kotlin.math.min

/** Converts point sequences into consistently sized trace images. */
class TracePreprocessing(
    private val randomFactory: () -> Random = { Random() }
) {
    private data class PreparedTrace(
        val points: List<Point>,
        val width: Int,
        val height: Int
    )

    fun medianThenNoise(points: List<Point>, sigma: Double = PRIVACY_SIGMA): List<Point> {
        if (points.size < 2) return points
        return addNoise(medianFilter(deduplicate(points)), sigma)
    }

    fun addNoise(points: List<Point>, sigma: Double): List<Point> {
        val random = randomFactory()
        return points.map { point ->
            Point(
                point.x + random.nextGaussian() * sigma,
                point.y + random.nextGaussian() * sigma
            )
        }
    }

    fun exportRawTraceForDataCollection(rawDataList: List<Point>): Bitmap =
        renderTrace(
            points = rawDataList,
            background = OPAQUE_BLACK,
            color = WHITE,
            useSpline = false,
            applyPrivacyFilter = false
        )

    fun exportRawTraceWithCvProcessing(rawDataList: List<Point>): Bitmap =
        renderTrace(
            points = rawDataList,
            background = TRANSPARENT,
            color = PRIVATE_TRACE_COLOR,
            useSpline = false,
            applyPrivacyFilter = true
        )

    fun exportSplineTraceWithCvProcessing(smoothDataList: List<Point>): Bitmap =
        renderTrace(
            points = smoothDataList,
            background = TRANSPARENT,
            color = PRIVATE_TRACE_COLOR,
            useSpline = true,
            applyPrivacyFilter = true
        )

    fun exportSplineTraceForDataCollection(smoothDataList: List<Point>): Bitmap =
        renderTrace(
            points = smoothDataList,
            background = OPAQUE_BLACK,
            color = WHITE,
            useSpline = true,
            applyPrivacyFilter = false
        )

    fun exportTrace(
        points: List<Point>,
        type: Settings.Trace.Type
    ): Bitmap? {
        if (points.size < type.minimumPoints) return null
        return when (type) {
            Settings.Trace.Type.RAW -> exportRawTraceForDataCollection(points)
            Settings.Trace.Type.RAW_CV -> exportRawTraceWithCvProcessing(points)
            Settings.Trace.Type.SPLINE -> exportSplineTraceForDataCollection(points)
            Settings.Trace.Type.SPLINE_CV -> exportSplineTraceWithCvProcessing(points)
        }
    }

    fun exportProcessedTraceForClass(
        points: List<Point>,
        type: Settings.Trace.Type
    ): Bitmap? = exportTrace(points, type)

    private fun renderTrace(
        points: List<Point>,
        background: Scalar,
        color: Scalar,
        useSpline: Boolean,
        applyPrivacyFilter: Boolean = false,
        fallBackToRaw: Boolean = false
    ): Bitmap {
        if (points.isEmpty()) return emptyBitmap(background)

        val prepared = prepare(points)
        val renderedPoints = if (applyPrivacyFilter) {
            medianThenNoise(prepared.points)
        } else {
            prepared.points
        }

        val mat = Mat(
            prepared.height,
            prepared.width,
            CvType.CV_8UC4,
            background
        )

        try {
            if (useSpline && renderedPoints.size >= MIN_SPLINE_POINTS) {
                TraceRenderer.drawSplineCurve(
                    renderedPoints,
                    mat,
                    color = color,
                    thickness = Settings.Trace.exportLineThickness
                )
            } else if (!useSpline || fallBackToRaw) {
                TraceRenderer.drawRawTrace(
                    renderedPoints,
                    mat,
                    color = color,
                    thickness = Settings.Trace.exportLineThickness
                )
            }
        } catch (error: Throwable) {
            mat.release()
            throw error
        }

        return mat.toScaledBitmap(prepared.width, prepared.height)
    }

    private fun prepare(points: List<Point>): PreparedTrace {
        val minX = points.minOf { it.x }
        val minY = points.minOf { it.y }
        val maxX = points.maxOf { it.x }
        val maxY = points.maxOf { it.y }
        val width = max(1, (maxX - minX + 2 * PADDING).toInt())
        val height = max(1, (maxY - minY + 2 * PADDING).toInt())
        val adjusted = points.map { point ->
            Point(point.x - minX + PADDING, point.y - minY + PADDING)
        }
        return PreparedTrace(adjusted, width, height)
    }

    private fun deduplicate(points: List<Point>): List<Point> = buildList {
        points.forEach { point ->
            if (isEmpty() || point != last()) add(point)
        }
    }

    private fun medianFilter(points: List<Point>): List<Point> =
        points.mapIndexed { index, _ ->
            val start = max(0, index - 1)
            val end = min(points.size, index + 2)
            val window = points.subList(start, end)
            val xCoordinates = window.map { it.x }.sorted()
            val yCoordinates = window.map { it.y }.sorted()
            Point(
                xCoordinates[xCoordinates.size / 2],
                yCoordinates[yCoordinates.size / 2]
            )
        }

    private fun Mat.toScaledBitmap(width: Int, height: Int): Bitmap {
        val intermediate = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        return try {
            Utils.matToBitmap(this, intermediate)
            Bitmap.createScaledBitmap(intermediate, OUTPUT_WIDTH, OUTPUT_HEIGHT, true).also { scaled ->
                if (scaled !== intermediate) intermediate.recycle()
            }
        } finally {
            release()
        }
    }

    private fun emptyBitmap(background: Scalar): Bitmap =
        Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply {
            eraseColor(if (background.`val`[3] == 0.0) Color.TRANSPARENT else Color.BLACK)
        }

    private companion object {
        const val OUTPUT_WIDTH = 79
        const val OUTPUT_HEIGHT = 68
        const val MIN_SPLINE_POINTS = 3
        const val PADDING = 30.0
        const val PRIVACY_SIGMA = 2.0
        val WHITE = Scalar(255.0, 255.0, 255.0, 255.0)
        val PRIVATE_TRACE_COLOR = Scalar(182.6, 173.2, 224.9, 255.0)
        val OPAQUE_BLACK = Scalar(0.0, 0.0, 0.0, 255.0)
        val TRANSPARENT = Scalar(0.0, 0.0, 0.0, 0.0)
    }
}
