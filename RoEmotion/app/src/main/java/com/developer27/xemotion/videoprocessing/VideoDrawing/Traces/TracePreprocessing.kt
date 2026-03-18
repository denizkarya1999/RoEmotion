package com.developer27.xemotion.videoprocessing.VideoDrawing.Traces

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

class TracePreprocessing {

    data class TraceVariants(
        val raw: Bitmap,
        val rawCv: Bitmap,
        val spline: Bitmap,
        val splineCv: Bitmap
    )

    // --------------------------------------------------
    // Applies median filtering and then Gaussian noise.
    // Used for privacy-preserving per-class trace output.
    // --------------------------------------------------
    fun medianThenNoise(points: List<Point>, sigma: Double = 2.0): List<Point> {
        if (points.size < 2) return points

        // Remove consecutive duplicate points first
        val dedup = mutableListOf<Point>().apply {
            points.forEach { p ->
                if (isEmpty() || p != last()) add(p)
            }
        }

        // Small sliding-window median filter
        val med = dedup.mapIndexed { i, _ ->
            val s = max(0, i - 1)
            val e = min(dedup.size, i + 2)
            val w = dedup.subList(s, e)

            val xs = w.map { it.x }.sorted()
            val ys = w.map { it.y }.sorted()

            Point(xs[xs.size / 2], ys[ys.size / 2])
        }

        // Add Gaussian noise for privacy
        val rng = Random()
        return med.map { p ->
            Point(
                p.x + rng.nextGaussian() * sigma,
                p.y + rng.nextGaussian() * sigma
            )
        }
    }

    /**
     * Adds zero-mean Gaussian noise to each point for privacy.
     */
    fun addNoise(points: List<Point>, sigma: Double): List<Point> {
        val rng = Random()

        return points.map { pt ->
            Point(
                pt.x + rng.nextGaussian() * sigma,
                pt.y + rng.nextGaussian() * sigma
            )
        }
    }

    // --------------------------------------------------------------------------------
    // Exports a raw trace bitmap for data collection / inference
    // --------------------------------------------------------------------------------
    fun exportRawTraceForDataCollection(rawDataList: List<Point>): Bitmap {
        val snapshot = rawDataList.toList()

        // Return a fallback bitmap if no points exist
        if (snapshot.isEmpty()) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.BLACK)
            }
        }

        // Find bounding box of the trace
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE

        for (pt in snapshot) {
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }

        val width = (maxX - minX).coerceAtLeast(1.0)
        val height = (maxY - minY).coerceAtLeast(1.0)
        val padding = 30.0

        val wDouble = width + 2.0 * padding
        val hDouble = height + 2.0 * padding
        val matWidth = max(1, wDouble.toInt())
        val matHeight = max(1, hDouble.toInt())

        // Black background image
        val mat = Mat(
            matHeight,
            matWidth,
            CvType.CV_8UC4,
            Scalar(0.0, 0.0, 0.0, 255.0)
        )

        // Shift points into local canvas space
        val adjustedPoints = snapshot.map {
            Point((it.x - minX) + padding, (it.y - minY) + padding)
        }

        // Temporarily override drawing settings
        val origColor = Settings.Trace.originalLineColor
        val origThickness = Settings.Trace.lineThickness

        Settings.Trace.originalLineColor = Scalar(255.0, 255.0, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawRawTrace(adjustedPoints, mat)

        // Restore original settings
        Settings.Trace.originalLineColor = origColor
        Settings.Trace.lineThickness = origThickness

        val intermediate = Bitmap.createBitmap(
            matWidth,
            matHeight,
            Bitmap.Config.ARGB_8888
        ).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }

        // Final export size
        return Bitmap.createScaledBitmap(intermediate, 79, 68, true)
    }

    /**
     * Exports the raw trace after median filtering and privacy noise.
     */
    fun exportRawTraceWithCvProcessing(rawDataList: List<Point>): Bitmap {
        val rawPoints = rawDataList.toList()

        // Transparent fallback bitmap
        if (rawPoints.isEmpty()) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.TRANSPARENT)
            }
        }

        // Compute bounding box
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE

        rawPoints.forEach { pt ->
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }

        val width = (maxX - minX).coerceAtLeast(1.0)
        val height = (maxY - minY).coerceAtLeast(1.0)
        val padding = 30.0

        val matW = max(1, (width + padding * 2).toInt())
        val matH = max(1, (height + padding * 2).toInt())

        // Transparent background image
        val mat = Mat(matH, matW, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 0.0))

        // Shift points into local export canvas
        val shiftedRaw = rawPoints.map { pt ->
            Point((pt.x - minX) + padding, (pt.y - minY) + padding)
        }

        // Remove consecutive duplicates
        val uniqueRaw = mutableListOf<Point>().apply {
            shiftedRaw.forEach { p ->
                if (isEmpty() || p != last()) add(p)
            }
        }

        // Median filter to suppress tremor-like small changes
        val medianFiltered = uniqueRaw.mapIndexed { i, _ ->
            val start = max(0, i - 1)
            val end = min(uniqueRaw.size, i + 2)
            val window = uniqueRaw.subList(start, end)

            val xs = window.map { it.x }.sorted()
            val ys = window.map { it.y }.sorted()

            Point(xs[xs.size / 2], ys[ys.size / 2])
        }

        // Add privacy noise
        val privateMedian = addNoise(medianFiltered, sigma = 2.0)

        // Temporarily change style for CV-processed output
        val prevColor = Settings.Trace.originalLineColor
        val prevTh = Settings.Trace.lineThickness

        Settings.Trace.originalLineColor = Scalar(182.6, 173.2, 224.9, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawRawTrace(privateMedian, mat)

        // Restore style
        Settings.Trace.originalLineColor = prevColor
        Settings.Trace.lineThickness = prevTh

        val intermediate = Bitmap.createBitmap(matW, matH, Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }

        return Bitmap.createScaledBitmap(intermediate, 79, 68, true)
    }

    /**
     * Exports the spline trace after median filtering and privacy noise.
     */
    fun exportSplineTraceWithCvProcessing(smoothDataList: List<Point>): Bitmap {
        val smoothPoints = smoothDataList.toList()

        // Transparent fallback bitmap
        if (smoothPoints.isEmpty()) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.TRANSPARENT)
            }
        }

        // Compute bounding box
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE

        smoothPoints.forEach { pt ->
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }

        val width = (maxX - minX).coerceAtLeast(1.0)
        val height = (maxY - minY).coerceAtLeast(1.0)
        val padding = 30.0

        val matW = max(1, (width + padding * 2).toInt())
        val matH = max(1, (height + padding * 2).toInt())
        val mat = Mat(matH, matW, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 0.0))

        // Shift points into export canvas
        val shifted = smoothPoints.map { pt ->
            Point((pt.x - minX) + padding, (pt.y - minY) + padding)
        }

        // Remove consecutive duplicates
        val unique = mutableListOf<Point>().apply {
            shifted.forEach { p ->
                if (isEmpty() || p != last()) add(p)
            }
        }

        // Median filtering
        val medianFiltered = unique.mapIndexed { i, _ ->
            val start = max(0, i - 1)
            val end = min(unique.size, i + 2)
            val window = unique.subList(start, end)

            val xs = window.map { it.x }.sorted()
            val ys = window.map { it.y }.sorted()

            Point(xs[xs.size / 2], ys[ys.size / 2])
        }

        // Add noise for privacy
        val noisy = addNoise(medianFiltered, sigma = 2.0)

        // Temporarily override spline rendering style
        val prevColor = Settings.Trace.splineLineColor
        val prevTh = Settings.Trace.lineThickness

        Settings.Trace.splineLineColor = Scalar(182.6, 173.2, 224.9, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawSplineCurve(noisy, mat)

        // Restore original style
        Settings.Trace.splineLineColor = prevColor
        Settings.Trace.lineThickness = prevTh

        val intermediate = Bitmap.createBitmap(matW, matH, Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }

        return Bitmap.createScaledBitmap(intermediate, 79, 68, true)
    }

    // --------------------------------------------------------------------------------
    // Exports a clean spline trace for data collection / inference
    // --------------------------------------------------------------------------------
    fun exportSplineTraceForDataCollection(smoothDataList: List<Point>): Bitmap {
        val snapshot = smoothDataList.toList()

        // Return fallback bitmap if no points are available
        if (snapshot.isEmpty()) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.BLACK)
            }
        }

        // Find trace bounds
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE

        for (pt in snapshot) {
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }

        val width = (maxX - minX).coerceAtLeast(1.0)
        val height = (maxY - minY).coerceAtLeast(1.0)
        val padding = 30.0

        val wDouble = width + 2.0 * padding
        val hDouble = height + 2.0 * padding
        val matWidth = max(1, wDouble.toInt())
        val matHeight = max(1, hDouble.toInt())

        // Black background image
        val mat = Mat(
            matHeight,
            matWidth,
            CvType.CV_8UC4,
            Scalar(0.0, 0.0, 0.0, 255.0)
        )

        // Shift points into local canvas
        val adjustedPoints = snapshot.map {
            Point((it.x - minX) + padding, (it.y - minY) + padding)
        }

        // Temporarily override spline style
        val origColor = Settings.Trace.splineLineColor
        val origThickness = Settings.Trace.lineThickness

        Settings.Trace.splineLineColor = Scalar(255.0, 255.0, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawSplineCurve(adjustedPoints, mat)

        // Restore original style
        Settings.Trace.splineLineColor = origColor
        Settings.Trace.lineThickness = origThickness

        val intermediate = Bitmap.createBitmap(
            matWidth,
            matHeight,
            Bitmap.Config.ARGB_8888
        ).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }

        return Bitmap.createScaledBitmap(intermediate, 79, 68, true)
    }

    /**
     * Builds a 79x68 processed spline bitmap from one user's raw trace.
     */
    fun exportProcessedTraceForClass(
        classId: Int,
        perClassRaw: Array<out MutableList<Point>>,
        perClassSmooth: Array<out MutableList<Point>>
    ): Bitmap? {
        val id = classId.coerceIn(0, 2)
        val rawPoints = perClassRaw[id].toList()

        if (rawPoints.isEmpty()) return null

        // Compute bounding box from raw points
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE

        rawPoints.forEach { p ->
            minX = min(minX, p.x)
            minY = min(minY, p.y)
            maxX = max(maxX, p.x)
            maxY = max(maxY, p.y)
        }

        val padding = 30.0
        val width = (maxX - minX).coerceAtLeast(1.0)
        val height = (maxY - minY).coerceAtLeast(1.0)
        val matW = max(1, (width + padding * 2).toInt())
        val matH = max(1, (height + padding * 2).toInt())

        // Shift into export coordinate system
        val shifted = rawPoints.map { p ->
            Point((p.x - minX) + padding, (p.y - minY) + padding)
        }

        // Remove duplicates
        val unique = mutableListOf<Point>().apply {
            shifted.forEach { q ->
                if (isEmpty() || q != last()) add(q)
            }
        }

        // Median + noise pipeline
        val processed: List<Point> = medianThenNoise(unique, sigma = 2.0)

        val mat = Mat(matH, matW, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 0.0))

        // Temporary rendering style
        val prevColor = Settings.Trace.splineLineColor
        val prevTh = Settings.Trace.lineThickness

        Settings.Trace.splineLineColor = Scalar(182.6, 173.2, 224.9, 255.0)
        Settings.Trace.lineThickness = 10

        // Use spline if enough points exist, otherwise raw polyline
        if (processed.size >= 3) {
            TraceRenderer.drawSplineCurve(processed, mat)
        } else {
            TraceRenderer.drawRawTrace(processed, mat)
        }

        // Restore original style
        Settings.Trace.splineLineColor = prevColor
        Settings.Trace.lineThickness = prevTh

        val intermediate = Bitmap.createBitmap(matW, matH, Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
            mat.release()
        }

        // Clear this user's stored trace after export
        perClassRaw[id].clear()
        perClassSmooth[id].clear()

        return Bitmap.createScaledBitmap(intermediate, 79, 68, true)
    }

    /**
     * Builds all 4 export variants for one user.
     * Does not clear the stored points.
     */
    fun exportPerUserTraceVariantsSnapshot(
        classId: Int,
        perClassRaw: Array<out MutableList<Point>>
    ): TraceVariants? {
        val id = classId.coerceIn(0, 2)
        val pts = perClassRaw[id].toList()

        if (pts.isEmpty()) return null

        // Compute bounds
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE

        pts.forEach { p ->
            minX = min(minX, p.x)
            minY = min(minY, p.y)
            maxX = max(maxX, p.x)
            maxY = max(maxY, p.y)
        }

        val width = (maxX - minX).coerceAtLeast(1.0)
        val height = (maxY - minY).coerceAtLeast(1.0)
        val padding = 30.0

        val matW = max(1, (width + padding * 2).toInt())
        val matH = max(1, (height + padding * 2).toInt())

        // Shift original points into export space
        val adjusted = pts.map {
            Point((it.x - minX) + padding, (it.y - minY) + padding)
        }

        // Helper to convert Mat -> Bitmap and resize
        fun toBitmapScaled(mat: Mat): Bitmap {
            val intermediate = Bitmap.createBitmap(matW, matH, Bitmap.Config.ARGB_8888).apply {
                Utils.matToBitmap(mat, this)
                mat.release()
            }
            return Bitmap.createScaledBitmap(intermediate, 79, 68, true)
        }

        // -------------------------
        // 1) Raw white-on-black
        // -------------------------
        val rawMat = Mat(matH, matW, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 255.0))

        val prevColor = Settings.Trace.originalLineColor
        val prevTh = Settings.Trace.lineThickness

        Settings.Trace.originalLineColor = Scalar(255.0, 255.0, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawRawTrace(adjusted, rawMat)
        val rawBmp = toBitmapScaled(rawMat)

        Settings.Trace.originalLineColor = prevColor
        Settings.Trace.lineThickness = prevTh

        // -------------------------
        // 2) Raw CV-processed
        // -------------------------
        val rawCvMat = Mat(matH, matW, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 0.0))
        val privatized = medianThenNoise(adjusted, sigma = 2.0)

        val prevColor2 = Settings.Trace.originalLineColor
        val prevTh2 = Settings.Trace.lineThickness

        Settings.Trace.originalLineColor = Scalar(182.6, 173.2, 224.9, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawRawTrace(privatized, rawCvMat)
        val rawCvBmp = toBitmapScaled(rawCvMat)

        Settings.Trace.originalLineColor = prevColor2
        Settings.Trace.lineThickness = prevTh2

        // -------------------------
        // 3) Spline white-on-black
        // -------------------------
        val splineMat = Mat(matH, matW, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 255.0))

        val prevSplineColor = Settings.Trace.splineLineColor
        val prevSplineTh = Settings.Trace.lineThickness

        Settings.Trace.splineLineColor = Scalar(255.0, 255.0, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawSplineCurve(adjusted, splineMat)
        val splineBmp = toBitmapScaled(splineMat)

        Settings.Trace.splineLineColor = prevSplineColor
        Settings.Trace.lineThickness = prevSplineTh

        // -------------------------
        // 4) Spline CV-processed
        // -------------------------
        val splineCvMat = Mat(matH, matW, CvType.CV_8UC4, Scalar(0.0, 0.0, 0.0, 0.0))

        val noisy = addNoise(
            run {
                // Remove consecutive duplicates first
                val unique = mutableListOf<Point>().apply {
                    adjusted.forEach { p ->
                        if (isEmpty() || p != last()) add(p)
                    }
                }

                // Median filter
                unique.mapIndexed { i, _ ->
                    val s = max(0, i - 1)
                    val e = min(unique.size, i + 2)
                    val w = unique.subList(s, e)

                    val xs = w.map { it.x }.sorted()
                    val ys = w.map { it.y }.sorted()

                    Point(xs[xs.size / 2], ys[ys.size / 2])
                }
            },
            sigma = 2.0
        )

        val prevSplineColor2 = Settings.Trace.splineLineColor
        val prevSplineTh2 = Settings.Trace.lineThickness

        Settings.Trace.splineLineColor = Scalar(182.6, 173.2, 224.9, 255.0)
        Settings.Trace.lineThickness = 10

        TraceRenderer.drawSplineCurve(noisy, splineCvMat)
        val splineCvBmp = toBitmapScaled(splineCvMat)

        Settings.Trace.splineLineColor = prevSplineColor2
        Settings.Trace.lineThickness = prevSplineTh2

        return TraceVariants(
            rawBmp,
            rawCvBmp,
            splineBmp,
            splineCvBmp
        )
    }
}