package com.developer27.xemotion.videoprocessing.CONTOUR

import android.graphics.Rect
import com.developer27.xemotion.videoprocessing.Settings
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc

// Contour detection utility
object ContourDetection {

    fun processContourDetection(mat: Mat): Triple<Point?, Rect?, Mat> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()

        try {
            Imgproc.findContours(
                mat.clone(),   // Use clone so findContours does not modify the original input
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
            )
        } finally {
            // Always release hierarchy mat
            hierarchy.release()
        }

        // Prepare output mat for drawing results
        val drawMat = Mat()
        if (mat.channels() == 1) {
            Imgproc.cvtColor(mat, drawMat, Imgproc.COLOR_GRAY2BGR)
        } else {
            mat.copyTo(drawMat)
        }

        // No contours found
        if (contours.isEmpty()) {
            return Triple(null, null, drawMat)
        }

        // Filter out very small contours if possible
        val minArea = 20.0
        val filteredContours = contours.filter { Imgproc.contourArea(it) >= minArea }

        // Pick the largest valid contour, or fallback to the largest overall contour
        val targetContour = (filteredContours.maxByOrNull { Imgproc.contourArea(it) }
            ?: contours.maxByOrNull { Imgproc.contourArea(it) })

        if (targetContour == null) {
            contours.forEach { it.release() }
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
        val cvRect = Imgproc.boundingRect(targetContour)
        val boundingRect = Rect(
            cvRect.x,
            cvRect.y,
            cvRect.x + cvRect.width,
            cvRect.y + cvRect.height
        )

        // Optionally draw the bounding box
        if (Settings.BoundingBox.enableBoundingBox) {
            Imgproc.rectangle(
                drawMat,
                cvRect,
                Settings.BoundingBox.boxColor,
                Settings.BoundingBox.boxThickness
            )
        }

        // Compute contour center from moments, fallback to rectangle center
        val moments = Imgproc.moments(targetContour)
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

        // Release contour memory
        contours.forEach { it.release() }

        return Triple(center, boundingRect, drawMat)
    }
}