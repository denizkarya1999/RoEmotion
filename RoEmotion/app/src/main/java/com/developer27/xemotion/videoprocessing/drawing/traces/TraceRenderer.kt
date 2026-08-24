package com.developer27.xemotion.videoprocessing.drawing.traces

import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.drawing.BoundingBox
import com.developer27.xemotion.videoprocessing.drawing.traces.patterns.KalmanBank
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.Random
import kotlin.math.max
import kotlin.math.min

// TraceRenderer - renders traces for raw and spline curves
object TraceRenderer {
    fun drawTrace(
        center: Point?,
        contourMat: Mat,
        rawDataList: MutableList<Point>,
        smoothDataList: MutableList<Point>
    ) {
        val (rawSnapshot, smoothSnapshot) = synchronized(rawDataList) {
            center?.let { detectedCenter ->
                rawDataList.add(detectedCenter)
                val (fx, fy) = KalmanBank.applyKalmanFilter(detectedCenter)
                smoothDataList.add(Point(fx, fy))
            }
            rawDataList.toList() to smoothDataList.toList()
        }

        with(Settings.Trace) {
            if (enableRAWtrace) {
                drawRawTrace(rawSnapshot, contourMat)
            }
            if (enableSPLINEtrace) {
                drawSplineCurve(smoothSnapshot, contourMat)
            }
        }
    }

    fun drawRawTrace(
        data: List<Point>,
        image: Mat,
        color: Scalar = Settings.Trace.originalLineColor,
        thickness: Int = Settings.Trace.lineThickness
    ) {
        for (i in 1 until data.size) {
            Imgproc.line(
                image,
                data[i - 1],
                data[i],
                color,
                thickness
            )
        }
    }

    fun drawSplineCurve(
        data: List<Point>,
        image: Mat,
        color: Scalar = Settings.Trace.splineLineColor,
        thickness: Int = Settings.Trace.lineThickness,
        step: Double = Settings.Trace.splineStep
    ) {
        if (data.size < 3) return

        val (splineX, splineY) = applySplineInterpolation(data)
        var prevPoint: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()
        val renderStep = boundedSplineStep(step, maxT)

        while (t <= maxT) {
            val currentPoint = Point(splineX.value(t), splineY.value(t))
            prevPoint?.let {
                Imgproc.line(
                    image,
                    it,
                    currentPoint,
                    color,
                    thickness
                )
            }
            prevPoint = currentPoint
            t += renderStep
        }
    }

    private fun applySplineInterpolation(data: List<Point>): Pair<PolynomialSplineFunction, PolynomialSplineFunction> {
        val interpolator = SplineInterpolator()
        val xData = data.map { it.x }.toDoubleArray()
        val yData = data.map { it.y }.toDoubleArray()
        val tData = data.indices.map { it.toDouble() }.toDoubleArray()

        val splineX = interpolator.interpolate(tData, xData)
        val splineY = interpolator.interpolate(tData, yData)

        return Pair(splineX, splineY)
    }

    // add offset-capable variants (so we can draw inside submat):
    private fun drawRawTraceOffset(
        data: List<Point>,
        image: Mat,
        offsetX: Double,
        offsetY: Double,
        color: Scalar,
        thickness: Int
    ) {
        for (i in 1 until data.size) {
            Imgproc.line(
                image,
                Point(data[i - 1].x - offsetX, data[i - 1].y - offsetY),
                Point(data[i].x - offsetX, data[i].y - offsetY),
                color,
                thickness
            )
        }
    }

    private fun drawSplineCurveOffset(
        data: List<Point>,
        image: Mat,
        offsetX: Double,
        offsetY: Double,
        color: Scalar,
        thickness: Int,
        step: Double
    ) {
        if (data.size < 3) return
        val (sx, sy) = applySplineInterpolation(data)
        var prev: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()
        val renderStep = boundedSplineStep(step, maxT)
        while (t <= maxT) {
            val pt = Point(sx.value(t) - offsetX, sy.value(t) - offsetY)
            prev?.let {
                Imgproc.line(image, it, pt, color, thickness)
            }
            prev = pt
            t += renderStep
        }
    }

    fun drawPathInBox(
        mat: Mat,
        box: BoundingBox,
        path: List<Point>,
        type: Settings.Trace.Type = Settings.Trace.inferenceType,
        color: Scalar = when (type) {
            Settings.Trace.Type.RAW,
            Settings.Trace.Type.RAW_CV -> Settings.Trace.originalLineColor
            Settings.Trace.Type.SPLINE,
            Settings.Trace.Type.SPLINE_CV -> Settings.Trace.splineLineColor
        },
        thickness: Int = Settings.Trace.lineThickness
    ) {
        if (path.size < 2) return

        // Clamp box (for gating only)
        val x1 = max(0, box.x1.toInt())
        val y1 = max(0, box.y1.toInt())
        val x2 = min(mat.cols(), box.x2.toInt())
        val y2 = min(mat.rows(), box.y2.toInt())
        if (x2 <= x1 || y2 <= y1) return

        // Mandatory overflow: draw on FULL frame (no ROI/submat).
        // Gate by a small expanded region so we only draw nearby strokes.
        val MARGIN_PX = 60
        val rx1 = (x1 - MARGIN_PX).coerceAtLeast(0)
        val ry1 = (y1 - MARGIN_PX).coerceAtLeast(0)
        val rx2 = (x2 + MARGIN_PX).coerceAtMost(mat.cols())
        val ry2 = (y2 + MARGIN_PX).coerceAtMost(mat.rows())
        fun nearBox(p: Point) = p.x >= rx1 && p.x <= rx2 && p.y >= ry1 && p.y <= ry2

        // --- CV processing: de-dupe -> median(3) -> Gaussian noise ---
        val dedup = mutableListOf<Point>().apply {
            path.forEach { p -> if (isEmpty() || p != last()) add(p) }
        }
        if (dedup.size < 2) return

        val applyCvProcessing = type == Settings.Trace.Type.RAW_CV ||
            type == Settings.Trace.Type.SPLINE_CV
        val processed = if (applyCvProcessing) {
            val medianFiltered = dedup.mapIndexed { index, _ ->
                val start = max(0, index - 1)
                val end = min(dedup.size, index + 2)
                val window = dedup.subList(start, end)
                val xCoordinates = window.map { it.x }.sorted()
                val yCoordinates = window.map { it.y }.sorted()
                Point(
                    xCoordinates[xCoordinates.size / 2],
                    yCoordinates[yCoordinates.size / 2]
                )
            }
            val random = Random()
            medianFiltered.map { point ->
                Point(
                    point.x + random.nextGaussian() * 2.0,
                    point.y + random.nextGaussian() * 2.0
                )
            }
        } else {
            dedup
        }

        // Keep only segments near the expanded box (but draw them on the full frame)
        val filtered = ArrayList<Point>(processed.size)
        for (i in 1 until processed.size) {
            val p0 = processed[i - 1]
            val p1 = processed[i]
            if (nearBox(p0) || nearBox(p1)) {
                if (filtered.isEmpty()) filtered.add(p0)
                filtered.add(p1)
            }
        }
        if (filtered.size < 2) return

        val useSpline = type == Settings.Trace.Type.SPLINE ||
            type == Settings.Trace.Type.SPLINE_CV
        if (!useSpline) {
            drawRawTrace(filtered, mat, color, thickness)
            return
        }

        // Draw the filtered path as a spline curve on the full image.
        if (filtered.size < 3) {
            // fallback: draw raw polyline if too few points for a spline
            for (i in 1 until filtered.size) {
                Imgproc.line(mat, filtered[i - 1], filtered[i], color, thickness)
            }
            return
        }

        // Use existing spline utilities
        val (sx, sy) = applySplineInterpolation(filtered)
        var prev: Point? = null
        var t = 0.0
        val tMax = (filtered.size - 1).toDouble()
        val step = boundedSplineStep(Settings.Trace.splineStep, tMax)
        while (t <= tMax) {
            val cur = Point(sx.value(t), sy.value(t))
            prev?.let { Imgproc.line(mat, it, cur, color, thickness) }
            prev = cur
            t += step
        }
    }

    private fun boundedSplineStep(requestedStep: Double, maxT: Double): Double {
        val safeRequested = requestedStep.takeIf { it.isFinite() && it > 0.0 } ?: DEFAULT_SPLINE_STEP
        return max(safeRequested, maxT / MAX_SPLINE_SEGMENTS)
    }

    private const val DEFAULT_SPLINE_STEP = 0.1
    private const val MAX_SPLINE_SEGMENTS = 1_000.0
}
