package com.developer27.xemotion.videoprocessing.contour

import android.graphics.Rect
import com.developer27.xemotion.videoprocessing.Settings
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc

// Contour detection utility
object ContourDetection {

    fun processContourDetection(mat: Mat): Triple<Point?, Rect?, Mat> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        val contourInput = mat.clone()
        val drawMat = Mat()
        var returnedDrawMat = false

        try {
            Imgproc.findContours(
                contourInput,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
            )
            // Prepare output mat for drawing results
            if (mat.channels() == 1) {
                Imgproc.cvtColor(mat, drawMat, Imgproc.COLOR_GRAY2BGR)
            } else {
                mat.copyTo(drawMat)
            }

            // No contours found
            if (contours.isEmpty()) {
                returnedDrawMat = true
                return Triple(null, null, drawMat)
            }

            // Filter out very small contours if possible
            val minArea = 20.0
            val filteredContours = contours.filter { Geometry.contourArea(it) >= minArea }

            // Pick the largest valid contour, or fallback to the largest overall contour
            val targetContour = filteredContours.maxByOrNull { Geometry.contourArea(it) }
                ?: contours.maxByOrNull { Geometry.contourArea(it) }
                ?: run {
                    returnedDrawMat = true
                    return Triple(null, null, drawMat)
                }

            // Draw the selected contour
            Imgproc.drawContours(
                drawMat,
                listOf(targetContour),
                -1,
                Settings.BoundingBox.boxColor,
                Settings.BoundingBox.boxThickness
            )

            // Compute bounding rectangle
            val cvRect = Geometry.boundingRect(targetContour)
            val boundingRect = Rect(
                cvRect.x,
                cvRect.y,
                cvRect.x + cvRect.width,
                cvRect.y + cvRect.height
            )

            // Compute contour center from moments, fallback to rectangle center
            val moments = Geometry.moments(targetContour)
            val center = if (moments.m00 != 0.0) {
                Point(
                    moments.m10 / moments.m00,
                    moments.m01 / moments.m00
                )
            } else {
                Point(
                    cvRect.x + cvRect.width / 2.0,
                    cvRect.y + cvRect.height / 2.0
                )
            }

            returnedDrawMat = true
            return Triple(center, boundingRect, drawMat)
        } finally {
            contourInput.release()
            hierarchy.release()
            contours.forEach(MatOfPoint::release)
            if (!returnedDrawMat) drawMat.release()
        }
    }
}
