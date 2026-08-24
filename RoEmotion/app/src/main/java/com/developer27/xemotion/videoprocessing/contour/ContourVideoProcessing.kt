package com.developer27.xemotion.videoprocessing.contour

import android.graphics.Bitmap
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.drawing.BoundingBox
import com.developer27.xemotion.videoprocessing.drawing.traces.TraceRenderer
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
    ): Bitmap {
        val pMat = Preprocessing.preprocessFrame(bitmap)
        var detectedMat: Mat? = null
        var drawMat: Mat? = null

        try {
            // 1) Detect contour and bounding region
            val detection = ContourDetection.processContourDetection(pMat)
            val center = detection.first
            val boundingRect = detection.second
            detectedMat = detection.third

            // Ensure output is always in BGR format for consistent drawing
            drawMat = ensureBgr(detectedMat)

            if (boundingRect != null) {
                val box = BoundingBox(
                    x1 = boundingRect.left.toFloat(),
                    y1 = boundingRect.top.toFloat(),
                    x2 = boundingRect.right.toFloat(),
                    y2 = boundingRect.bottom.toFloat(),
                    confidence = 1f,
                    classId = 0
                )

                val labelText = buildCombinedLabel(
                    userLabel = userLabel,
                    classificationLabel = classificationLabel
                )

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

                if (Settings.BoundingBox.enableBoundingBox && labelText.isNotBlank()) {
                    drawTopLabel(
                        drawMat,
                        box,
                        labelText,
                        Scalar(255.0, 255.0, 255.0)
                    )
                }
            }

            // 2) Draw trace (raw + smoothed points)
            TraceRenderer.drawTrace(center, drawMat, rawDataList, smoothDataList)

            // 3) Convert processed Mat back to Bitmap
            val overlay = Bitmap.createBitmap(drawMat.cols(), drawMat.rows(), Bitmap.Config.ARGB_8888)
            return try {
                Utils.matToBitmap(drawMat, overlay)
                overlay
            } catch (error: Throwable) {
                overlay.recycle()
                throw error
            }
        } finally {
            pMat.release()
            detectedMat?.release()
            if (drawMat !== detectedMat) drawMat?.release()
        }
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

}

// --------------------------------------------------
// Frame preprocessing pipeline
// --------------------------------------------------
object Preprocessing {

    fun preprocessFrame(src: Bitmap): Mat {
        val source = Mat()
        val grayscale = Mat()
        val enhanced = Mat()
        val thresholded = Mat()
        val blurred = Mat()
        val output = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))

        try {
            Utils.bitmapToMat(src, source)
            Imgproc.cvtColor(source, grayscale, Imgproc.COLOR_BGR2GRAY)
            Core.multiply(grayscale, Scalar(Settings.Brightness.factor), enhanced)
            Imgproc.threshold(
                enhanced,
                thresholded,
                Settings.Brightness.threshold,
                255.0,
                Imgproc.THRESH_TOZERO
            )
            Imgproc.GaussianBlur(thresholded, blurred, Size(5.0, 5.0), 0.0)
            Imgproc.morphologyEx(blurred, output, Imgproc.MORPH_CLOSE, kernel)
            return output
        } catch (error: Throwable) {
            output.release()
            throw error
        } finally {
            source.release()
            grayscale.release()
            enhanced.release()
            thresholded.release()
            blurred.release()
            kernel.release()
        }
    }
}
