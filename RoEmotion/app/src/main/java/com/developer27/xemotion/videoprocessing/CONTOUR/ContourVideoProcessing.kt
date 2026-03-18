package com.developer27.xemotion.videoprocessing.CONTOUR

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoDrawing.BoundingBox
import com.developer27.xemotion.videoprocessing.VideoDrawing.Traces.TraceRenderer
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class ContourVideoProcessing(
    private val drawTopLabel: (Mat, BoundingBox, String, Scalar) -> Unit,
    private val rawDataList: MutableList<Point>,
    private val smoothDataList: MutableList<Point>
) {

    fun processFrame(
        bitmap: Bitmap,
        userLabel: String,
        classificationLabel: String
    ): Pair<Bitmap, Bitmap>? {

        // 1) Preprocess frame (grayscale, thresholding, blur, morphology)
        val (pMat, _) = Preprocessing.preprocessFrame(bitmap)

        // 2) Detect contour and bounding region
        val (center, boundingRect, detectedMat) = ContourDetection.processContourDetection(pMat)

        // Ensure output is always in BGR format for consistent drawing
        val drawMat = ensureBgr(detectedMat)

        if (boundingRect != null) {
            val box = BoundingBox(
                x1 = boundingRect.left.toFloat(),
                y1 = boundingRect.top.toFloat(),
                x2 = boundingRect.right.toFloat(),
                y2 = boundingRect.bottom.toFloat(),
                confidence = 1f,
                classId = 0
            )

            // Combine user + classification label into one string
            val labelText = buildCombinedLabel(
                userLabel = userLabel,
                classificationLabel = classificationLabel
            )

            // Draw bounding box if enabled and valid
            if (Settings.BoundingBox.enableBoundingBox &&
                box.x2 > box.x1 &&
                box.y2 > box.y1
            ) {
                Imgproc.rectangle(
                    drawMat,
                    Point(box.x1.toDouble(), box.y1.toDouble()),
                    Point(box.x2.toDouble(), box.y2.toDouble()),
                    Settings.BoundingBox.boxColor,
                    Settings.BoundingBox.boxThickness
                )
            }

            // Draw label above bounding box if available
            if (labelText.isNotBlank()) {
                drawTopLabel(
                    drawMat,
                    box,
                    labelText,
                    Scalar(255.0, 255.0, 255.0)
                )
            }
        }

        // 3) Draw trace (raw + smoothed points)
        TraceRenderer.drawTrace(center, drawMat, rawDataList, smoothDataList)

        // 4) Convert processed Mat back to Bitmap
        val overlay = Bitmap.createBitmap(drawMat.cols(), drawMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(drawMat, overlay)

        // 5) Extract cropped region if bounding box exists
        val cropped: Bitmap = boundingRect
            ?.let { extractBoundingBoxRegion(bitmap, it) }
            ?: overlay

        // Release OpenCV memory
        pMat.release()
        detectedMat.release()
        if (drawMat !== detectedMat) {
            drawMat.release()
        }

        return overlay to cropped
    }

    // Builds final label text shown above bounding box
    private fun buildCombinedLabel(
        userLabel: String,
        classificationLabel: String
    ): String {
        val safeUser = userLabel.trim()
        val safeEmotion = classificationLabel.trim()

        return when {
            safeUser.isNotBlank() && safeEmotion.isNotBlank() -> "$safeUser | $safeEmotion"
            safeEmotion.isNotBlank() -> safeEmotion
            safeUser.isNotBlank() -> safeUser
            else -> "Contour"
        }
    }

    // Ensures Mat is in BGR format (required for consistent OpenCV drawing)
    private fun ensureBgr(src: Mat): Mat {
        return when (src.channels()) {
            3 -> src // already BGR
            1 -> {
                val bgr = Mat()
                Imgproc.cvtColor(src, bgr, Imgproc.COLOR_GRAY2BGR)
                bgr
            }
            4 -> {
                val bgr = Mat()
                Imgproc.cvtColor(src, bgr, Imgproc.COLOR_RGBA2BGR)
                bgr
            }
            else -> {
                val bgr = Mat()
                src.convertTo(bgr, CvType.CV_8UC3)
                bgr
            }
        }
    }

    // Crops the bounding box region from the original bitmap
    private fun extractBoundingBoxRegion(srcBitmap: Bitmap, boundingRect: Rect): Bitmap {
        val left = boundingRect.left.coerceAtLeast(0)
        val top = boundingRect.top.coerceAtLeast(0)
        val right = boundingRect.right.coerceAtMost(srcBitmap.width)
        val bottom = boundingRect.bottom.coerceAtMost(srcBitmap.height)

        // Return fallback bitmap if invalid region
        if (left >= right || top >= bottom) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.BLACK)
            }
        }

        val width = right - left
        val height = bottom - top

        return Bitmap.createBitmap(srcBitmap, left, top, width, height)
    }
}

// --------------------------------------------------
// Frame preprocessing pipeline
// --------------------------------------------------
object Preprocessing {

    fun preprocessFrame(src: Bitmap): Pair<Mat, Bitmap> {

        // Convert Bitmap → OpenCV Mat
        val sMat = Mat().also { Utils.bitmapToMat(src, it) }

        // Convert to grayscale
        val gMat = Mat().also {
            Imgproc.cvtColor(sMat, it, Imgproc.COLOR_BGR2GRAY)
            sMat.release()
        }

        // Apply brightness scaling
        val eMat = Mat().also {
            Core.multiply(gMat, Scalar(Settings.Brightness.factor), it)
            gMat.release()
        }

        // Thresholding to remove low-intensity pixels
        val tMat = Mat().also {
            Imgproc.threshold(
                eMat,
                it,
                Settings.Brightness.threshold,
                255.0,
                Imgproc.THRESH_TOZERO
            )
            eMat.release()
        }

        // Smooth image using Gaussian blur
        val bMat = Mat().also {
            Imgproc.GaussianBlur(tMat, it, Size(5.0, 5.0), 0.0)
            tMat.release()
        }

        // Morphological closing to fill small gaps
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val cMat = Mat().also {
            Imgproc.morphologyEx(bMat, it, Imgproc.MORPH_CLOSE, k)
            bMat.release()
        }

        // Convert processed Mat back to Bitmap (optional output)
        val bmp = Bitmap.createBitmap(cMat.cols(), cMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(cMat, bmp)

        return cMat to bmp
    }
}