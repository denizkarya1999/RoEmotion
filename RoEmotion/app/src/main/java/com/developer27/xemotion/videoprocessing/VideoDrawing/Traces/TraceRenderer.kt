package com.developer27.xemotion.videoprocessing.VideoDrawing.Traces

import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoDrawing.BoundingBox
import com.developer27.xemotion.videoprocessing.VideoDrawing.Traces.TracePatterns.KalmanBank
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
        center?.let { detectedCenter ->
            rawDataList.add(detectedCenter)
            val (fx, fy) = KalmanBank.applyKalmanFilter(detectedCenter)
            smoothDataList.add(Point(fx, fy))
        }

        with(Settings.Trace) {
            if (enableRAWtrace) {
                drawRawTrace(rawDataList, contourMat)
            }
            if (enableSPLINEtrace) {
                drawSplineCurve(smoothDataList, contourMat)
            }
        }
    }

    fun drawRawTrace(data: List<Point>, image: Mat) {
        for (i in 1 until data.size) {
            Imgproc.line(
                image,
                data[i - 1],
                data[i],
                Settings.Trace.originalLineColor,
                Settings.Trace.lineThickness
            )
        }
    }

    fun drawSplineCurve(data: List<Point>, image: Mat) {
        if (data.size < 3) return

        val (splineX, splineY) = applySplineInterpolation(data)
        var prevPoint: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()

        while (t <= maxT) {
            val currentPoint = Point(splineX.value(t), splineY.value(t))
            prevPoint?.let {
                Imgproc.line(
                    image,
                    it,
                    currentPoint,
                    Settings.Trace.splineLineColor,
                    Settings.Trace.lineThickness
                )
            }
            prevPoint = currentPoint
            t += Settings.Trace.splineStep
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
        while (t <= maxT) {
            val pt = Point(sx.value(t) - offsetX, sy.value(t) - offsetY)
            prev?.let {
                Imgproc.line(image, it, pt, color, thickness)
            }
            prev = pt
            t += step
        }
    }

    fun drawPathInBox(
        mat: Mat,
        box: BoundingBox,
        path: List<Point>,
        color: Scalar = Settings.Trace.splineLineColor,  // use spline color
        thickness: Int = Settings.Trace.lineThickness
    ) {
        // If no writing is enabled, do nothing.
        if (!(Settings.Trace.enableRAWtrace || Settings.Trace.enableSPLINEtrace)) return
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

        val medianFiltered = dedup.mapIndexed { i, _ ->
            val s = max(0, i - 1);
            val e = min(dedup.size, i + 2)
            val win = dedup.subList(s, e)
            val xs = win.map { it.x }.sorted();
            val ys = win.map { it.y }.sorted()
            Point(xs[xs.size / 2], ys[ys.size / 2])
        }
        val noisy = run {
            val rng = Random()
            val sigma = 2.0
            medianFiltered.map { p ->
                Point(
                    p.x + rng.nextGaussian() * sigma,
                    p.y + rng.nextGaussian() * sigma
                )
            }
        }

        // Keep only segments near the expanded box (but draw them on the full frame)
        val filtered = ArrayList<Point>(noisy.size)
        for (i in 1 until noisy.size) {
            val p0 = noisy[i - 1];
            val p1 = noisy[i]
            if (nearBox(p0) || nearBox(p1)) {
                if (filtered.isEmpty()) filtered.add(p0)
                filtered.add(p1)
            }
        }
        if (filtered.size < 2) return

        // --- Draw as a spline curve on the full image ---
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
        val step = Settings.Trace.splineStep
        val tMax = (filtered.size - 1).toDouble()
        while (t <= tMax) {
            val cur = Point(sx.value(t), sy.value(t))
            prev?.let { Imgproc.line(mat, it, cur, color, thickness) }
            prev = cur
            t += step
        }
    }
}