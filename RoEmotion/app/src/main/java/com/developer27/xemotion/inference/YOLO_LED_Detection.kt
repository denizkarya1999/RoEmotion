package com.developer27.xemotion.inference

import android.graphics.Bitmap
import android.util.Log
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoDrawing.BoundingBox
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.collections.plusAssign
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

// --------------------------------------------------
// YOLO LED detection output format from Ultralytics TFLite:
// [1, 300, 6]
// --------------------------------------------------
data class YoloDet(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val objConf: Float,
    val classId: Int,
    val classScore: Float
)

data class LetterboxMeta(
    val inputBitmap: Bitmap,
    val scale: Float,
    val padLeft: Int,
    val padTop: Int,
    val resizedWidth: Int,
    val resizedHeight: Int,
    val targetWidth: Int,
    val targetHeight: Int
)

object YOLO_LED_Detection {

    private const val TAG = "YOLO_LED_Detection"
    private const val EXPECTED_CLASSES = 3

    // Maps class indices to User_1, User_2, User_3
    private val classNumbers = IntArray(EXPECTED_CLASSES) { it + 1 }
    val userLabels = Array(EXPECTED_CLASSES) { idx -> "User_${classNumbers[idx]}" }

    // Controls whether prediction logs are printed
    private fun shouldLog(): Boolean = Settings.ExportData.enablePredictionLogging

    // Safe sigmoid with clamping for numerical stability
    private fun sigmoid(x: Float): Float {
        if (!x.isFinite()) return 0f
        val clamped = x.coerceIn(-30f, 30f)
        return (1.0f / (1.0f + exp(-clamped)))
    }

    // Ensures scores stay in [0, 1]
    private fun sanitizeProbability(x: Float): Float {
        if (!x.isFinite()) return 0f
        return when {
            x < 0f || x > 1f -> sigmoid(x)
            else -> x
        }.coerceIn(0f, 1f)
    }

    // Replaces invalid coordinates with 0
    private fun sanitizeCoord(x: Float): Float {
        return if (x.isFinite()) x else 0f
    }

    // Logs per-box class score information
    private fun logClassScores(boxIdx: Int, classId: Int, score: Float) {
        if (!shouldLog()) return

        val scores = FloatArray(EXPECTED_CLASSES) { 0f }
        if (classId in 0 until EXPECTED_CLASSES) {
            scores[classId] = score
        }

        Log.d(
            TAG,
            "Box#$boxIdx -> " +
                    "User_1=${"%.4f".format(Locale.US, scores[0])}, " +
                    "User_2=${"%.4f".format(Locale.US, scores[1])}, " +
                    "User_3=${"%.4f".format(Locale.US, scores[2])}"
        )
    }

    // Creates a letterboxed bitmap and returns resize/padding metadata
    fun createLetterboxedBitmap(
        srcBitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        padColor: Scalar = Scalar(114.0, 114.0, 114.0)
    ): LetterboxMeta {
        val srcW = srcBitmap.width
        val srcH = srcBitmap.height

        // Uniform scaling while preserving aspect ratio
        val scale = min(
            targetWidth.toFloat() / srcW.toFloat(),
            targetHeight.toFloat() / srcH.toFloat()
        )

        val resizedW = (srcW * scale).roundToInt().coerceAtLeast(1)
        val resizedH = (srcH * scale).roundToInt().coerceAtLeast(1)

        val padW = targetWidth - resizedW
        val padH = targetHeight - resizedH

        // Center padding offsets
        val padLeft = padW / 2
        val padTop = padH / 2

        val srcMat = Mat().also { Utils.bitmapToMat(srcBitmap, it) }

        // Resize source to fit target
        val resized = Mat()
        Imgproc.resize(srcMat, resized, Size(resizedW.toDouble(), resizedH.toDouble()))
        srcMat.release()

        // Create final padded image
        val out = Mat(targetHeight, targetWidth, resized.type(), padColor)

        val roi = out.submat(Rect(padLeft, padTop, resizedW, resizedH))
        resized.copyTo(roi)
        roi.release()
        resized.release()

        val outputBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(out, outputBitmap)
        out.release()

        return LetterboxMeta(
            inputBitmap = outputBitmap,
            scale = scale,
            padLeft = padLeft,
            padTop = padTop,
            resizedWidth = resizedW,
            resizedHeight = resizedH,
            targetWidth = targetWidth,
            targetHeight = targetHeight
        )
    }

    // Converts bitmap to NHWC float tensor normalized to [0, 1]
    fun bitmapToNormalizedTensorNHWC(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height

        val inputBuffer = ByteBuffer.allocateDirect(4 * width * height * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var pixelIndex = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val px = pixels[pixelIndex++]

                val r = ((px shr 16) and 0xFF) / 255.0f
                val g = ((px shr 8) and 0xFF) / 255.0f
                val b = (px and 0xFF) / 255.0f

                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            }
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    // Parses raw TFLite output into detections and applies per-class NMS
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): List<YoloDet> {
        if (rawOutput.isEmpty() || rawOutput[0].isEmpty() || rawOutput[0][0].isEmpty()) {
            return emptyList()
        }

        val numBoxes = rawOutput[0].size
        val elemsPerBox = rawOutput[0][0].size

        val detections = ArrayList<YoloDet>()

        // Expected output element count: [x1, y1, x2, y2, score, class]
        if (elemsPerBox != 6) {
            return emptyList()
        }

        for (i in 0 until numBoxes) {
            val det = rawOutput[0][i]

            val x1 = sanitizeCoord(det[0])
            val y1 = sanitizeCoord(det[1])
            val x2 = sanitizeCoord(det[2])
            val y2 = sanitizeCoord(det[3])
            val score = sanitizeProbability(det[4])

            val rawClass = det[5]
            val classId = if (rawClass.isFinite()) rawClass.toInt() else -1

            logClassScores(i, classId, score)

            val w = x2 - x1
            val h = y2 - y1

            // Basic filtering
            if (score < Settings.Inference.confidenceThreshold) continue
            if (w <= 0f || h <= 0f) continue
            if (classId !in 0 until EXPECTED_CLASSES) continue

            detections += YoloDet(
                xCenter = (x1 + x2) / 2f,
                yCenter = (y1 + y2) / 2f,
                width = w,
                height = h,
                objConf = score,
                classId = classId,
                classScore = score
            )
        }

        if (detections.isEmpty()) return emptyList()

        // Sort by confidence before NMS
        val boxes = detections
            .map { det -> det to detectionToBox(det) }
            .sortedByDescending { it.first.objConf }

        val picked = ArrayList<YoloDet>()
        val mutable = boxes.toMutableList()

        // Per-class NMS
        while (mutable.isNotEmpty()) {
            val (winnerDet, winnerBox) = mutable.removeAt(0)
            picked += winnerDet

            mutable.removeAll { other ->
                (other.first.classId == winnerDet.classId) &&
                        (computeIoU(winnerBox, other.second) > Settings.Inference.iouThreshold)
            }
        }

        return picked
    }

    // Keeps only the best detection for each class
    fun pickOnePerClass(dets: List<YoloDet>, nClasses: Int = EXPECTED_CLASSES): List<YoloDet> {
        if (dets.isEmpty()) return emptyList()

        // Fallback if class IDs are invalid
        if (dets.any { it.classId < 0 }) {
            return dets
                .sortedByDescending { it.objConf }
                .take(nClasses)
        }

        val bestPerClass = mutableMapOf<Int, YoloDet>()

        for (d in dets) {
            if (d.classId in 0 until nClasses) {
                val current = bestPerClass[d.classId]
                if (current == null || d.objConf > current.objConf) {
                    bestPerClass[d.classId] = d
                }
            }
        }

        return (0 until nClasses).mapNotNull { bestPerClass[it] }
    }

    // Converts YoloDet into BoundingBox format
    private fun detectionToBox(d: YoloDet) = BoundingBox(
        x1 = d.xCenter - d.width / 2f,
        y1 = d.yCenter - d.height / 2f,
        x2 = d.xCenter + d.width / 2f,
        y2 = d.yCenter + d.height / 2f,
        confidence = d.objConf,
        classId = if (d.classId >= 0) d.classId else 0
    )

    // Computes IoU between two bounding boxes
    private fun computeIoU(boxA: BoundingBox, boxB: BoundingBox): Float {
        val x1 = max(boxA.x1, boxB.x1)
        val y1 = max(boxA.y1, boxB.y1)
        val x2 = min(boxA.x2, boxB.x2)
        val y2 = min(boxA.y2, boxB.y2)

        val intersectionW = max(0f, x2 - x1)
        val intersectionH = max(0f, y2 - y1)
        val intersectionArea = intersectionW * intersectionH

        val areaA = max(0f, boxA.x2 - boxA.x1) * max(0f, boxA.y2 - boxA.y1)
        val areaB = max(0f, boxB.x2 - boxB.x1) * max(0f, boxB.y2 - boxB.y1)
        val unionArea = areaA + areaB - intersectionArea

        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }

    // Maps detections from model/letterbox space back to original image space
    fun rescaleDetections(
        detections: List<YoloDet>,
        originalWidth: Int,
        originalHeight: Int,
        padOffsets: Pair<Int, Int>,
        modelInputWidth: Int,
        modelInputHeight: Int,
        scale: Float? = null
    ): List<Pair<BoundingBox, Point>> {
        if (detections.isEmpty()) return emptyList()

        val padLeft = padOffsets.first.toFloat()
        val padTop = padOffsets.second.toFloat()

        // Reuse provided scale or recompute it
        val gain = scale ?: min(
            modelInputWidth.toFloat() / originalWidth.toFloat(),
            modelInputHeight.toFloat() / originalHeight.toFloat()
        )

        return detections.mapNotNull { d ->
            // Handles outputs that may already be normalized
            val looksNormalized =
                d.xCenter in 0f..1.5f &&
                        d.yCenter in 0f..1.5f &&
                        d.width in 0f..1.5f &&
                        d.height in 0f..1.5f

            val xCenterModel = if (looksNormalized) d.xCenter * modelInputWidth else d.xCenter
            val yCenterModel = if (looksNormalized) d.yCenter * modelInputHeight else d.yCenter
            val wModel = if (looksNormalized) d.width * modelInputWidth else d.width
            val hModel = if (looksNormalized) d.height * modelInputHeight else d.height

            val x1Model = xCenterModel - wModel / 2f
            val y1Model = yCenterModel - hModel / 2f
            val x2Model = xCenterModel + wModel / 2f
            val y2Model = yCenterModel + hModel / 2f

            // Remove padding and rescale back to original image
            val x1 = (x1Model - padLeft) / gain
            val y1 = (y1Model - padTop) / gain
            val x2 = (x2Model - padLeft) / gain
            val y2 = (y2Model - padTop) / gain

            // Clamp to valid image boundaries
            val clampedX1 = x1.coerceIn(0f, originalWidth.toFloat())
            val clampedY1 = y1.coerceIn(0f, originalHeight.toFloat())
            val clampedX2 = x2.coerceIn(0f, originalWidth.toFloat())
            val clampedY2 = y2.coerceIn(0f, originalHeight.toFloat())

            val box = BoundingBox(
                x1 = min(clampedX1, clampedX2),
                y1 = min(clampedY1, clampedY2),
                x2 = max(clampedX1, clampedX2),
                y2 = max(clampedY1, clampedY2),
                confidence = d.objConf,
                classId = maxOf(d.classId, 0)
            )

            // Also return box center point
            val center = Point(
                ((box.x1 + box.x2) / 2f).toDouble(),
                ((box.y1 + box.y2) / 2f).toDouble()
            )

            box to center
        }
    }
}