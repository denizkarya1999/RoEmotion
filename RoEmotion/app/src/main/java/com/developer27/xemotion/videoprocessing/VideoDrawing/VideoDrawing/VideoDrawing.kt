package com.developer27.xemotion.videoprocessing.VideoDrawing

import com.developer27.xemotion.videoprocessing.Settings
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

// --------------------------------------------------
// Bounding box representation (YOLO-style)
// --------------------------------------------------
data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
)

class VideoDrawingHelper(
    private val perUserEmotion: Array<String>,     // Emotion label per user
    private val perUserColor: Array<Scalar>,       // Color per user (or emotion)
    private val perUserConfidence: FloatArray      // Emotion confidence per user
) {

    companion object {
        private const val FONT_SCALE = 2.0
        private const val THICKNESS = 3
        private const val TEXT_MARGIN = 8
        private const val LABEL_PADDING = 6
    }

    // Ensures bounding box stays within frame boundaries
    private fun clampBoxToFrame(mat: Mat, box: BoundingBox): BoundingBox {
        val maxX = (mat.cols() - 1).coerceAtLeast(0).toFloat()
        val maxY = (mat.rows() - 1).coerceAtLeast(0).toFloat()

        val left = box.x1.coerceIn(0f, maxX)
        val top = box.y1.coerceIn(0f, maxY)
        val right = box.x2.coerceIn(0f, maxX)
        val bottom = box.y2.coerceIn(0f, maxY)

        return BoundingBox(
            x1 = min(left, right),
            y1 = min(top, bottom),
            x2 = max(left, right),
            y2 = max(top, bottom),
            confidence = box.confidence,
            classId = box.classId
        )
    }

    // Draws a label above (or near) the bounding box
    fun drawTopLabel(mat: Mat, box: BoundingBox, text: String, color: Scalar) {
        if (text.isBlank()) return
        if (mat.empty() || mat.cols() <= 0 || mat.rows() <= 0) return

        val safeBox = clampBoxToFrame(mat, box)

        // Compute text size
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(
            text,
            Imgproc.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            THICKNESS,
            baseline
        )

        val rawTextWidth = textSize.width.toInt()
        val textHeight = textSize.height.toInt()
        val base = baseline[0]

        val frameWidth = mat.cols()
        val frameHeight = mat.rows()

        // Label box dimensions with padding
        val labelWidthRaw = rawTextWidth + LABEL_PADDING * 2
        val labelHeightRaw = textHeight + base + LABEL_PADDING * 2

        val labelWidth = labelWidthRaw.coerceAtMost(frameWidth)
        val labelHeight = labelHeightRaw.coerceAtMost(frameHeight)

        // Position label horizontally near box left
        val preferredX = safeBox.x1.toInt()
        val maxLabelX = (frameWidth - labelWidth).coerceAtLeast(0)
        val labelX = preferredX.coerceIn(0, maxLabelX)

        // Prefer placing label above the box
        val preferredAboveY = (safeBox.y1.toInt() - labelHeight - TEXT_MARGIN)
        val maxLabelY = (frameHeight - labelHeight).coerceAtLeast(0)

        val labelY = if (preferredAboveY >= 0) {
            preferredAboveY.coerceIn(0, maxLabelY)
        } else {
            // fallback: place inside/at top of box
            safeBox.y1.toInt().coerceIn(0, maxLabelY)
        }

        val rectLeft = labelX
        val rectTop = labelY
        val rectRight = (labelX + labelWidth).coerceAtMost(frameWidth)
        val rectBottom = (labelY + labelHeight).coerceAtMost(frameHeight)

        if (rectRight <= rectLeft || rectBottom <= rectTop) return

        // Draw black background rectangle for readability
        Imgproc.rectangle(
            mat,
            Point(rectLeft.toDouble(), rectTop.toDouble()),
            Point(rectRight.toDouble(), rectBottom.toDouble()),
            Scalar(0.0, 0.0, 0.0),
            Imgproc.FILLED
        )

        // Compute text origin inside the label box
        val textX = (rectLeft + LABEL_PADDING).coerceAtMost((frameWidth - 1).coerceAtLeast(0))
        val textY = (rectBottom - LABEL_PADDING - base).coerceIn(
            0,
            (frameHeight - 1).coerceAtLeast(0)
        )

        // Draw label text
        Imgproc.putText(
            mat,
            text,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            color,
            THICKNESS
        )
    }

    // Draws bounding box + combined user/emotion label
    fun drawUserBoxAndLabel(mat: Mat, box: BoundingBox, classId: Int) {
        if (mat.empty() || mat.cols() <= 0 || mat.rows() <= 0) return

        val safeClassId = classId.coerceIn(0, 2)
        val safeBox = clampBoxToFrame(mat, box)

        // Skip invalid boxes
        if (safeBox.x2 <= safeBox.x1 || safeBox.y2 <= safeBox.y1) return

        // Select color based on current mode
        val borderColor = when (Settings.BoundingBox.colorMode) {
            Settings.BoundingBox.ColorMode.BY_USER -> perUserColor[safeClassId]
            Settings.BoundingBox.ColorMode.BY_EMOTION -> perUserColor[safeClassId]
        }

        // Draw bounding box
        Imgproc.rectangle(
            mat,
            Point(safeBox.x1.toDouble(), safeBox.y1.toDouble()),
            Point(safeBox.x2.toDouble(), safeBox.y2.toDouble()),
            borderColor,
            Settings.BoundingBox.boxThickness
        )

        val shortUser = "User_${safeClassId + 1}"

        // LED detection confidence (from YOLO)
        val ledConfPct = (safeBox.confidence * 100f).coerceIn(0f, 100f)
        val ledConfStr = String.format(Locale.US, "%.1f%%", ledConfPct)

        // Emotion + confidence
        val emotion = perUserEmotion[safeClassId].trim()
        val emotionConf = perUserConfidence[safeClassId]
        val emotionConfStr =
            if (!emotionConf.isNaN()) String.format(Locale.US, "%.1f%%", emotionConf) else ""

        // Build final label string
        val label = when {
            emotion.isBlank() -> "$shortUser ($ledConfStr)"
            emotionConfStr.isBlank() -> "$shortUser ($ledConfStr) | $emotion"
            else -> "$shortUser ($ledConfStr) | $emotion ($emotionConfStr)"
        }

        // Draw label above box
        drawTopLabel(mat, safeBox, label, borderColor)
    }
}