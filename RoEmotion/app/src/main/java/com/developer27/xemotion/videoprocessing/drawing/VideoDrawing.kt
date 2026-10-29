package com.developer27.xemotion.videoprocessing.drawing

import com.developer27.xemotion.videoprocessing.Settings
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
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

data class UserEmotionDisplay(
    val emotion: String,
    val color: Scalar,
    val confidence: Float
)

class VideoDrawingHelper(
    private val emotionDisplayForUser: (Int) -> UserEmotionDisplay
) {

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

    // Draws a label above (or near) the bounding box.
    fun drawTopLabel(mat: Mat, box: BoundingBox, text: String, color: Scalar,
                     style: OverlayStyle = OverlayStyle.HANDHELD) {
        drawLabel(mat, box, text, color, placeBelow = false, style = style)
    }

    private fun drawLabel(
        mat: Mat,
        box: BoundingBox,
        text: String,
        color: Scalar,
        placeBelow: Boolean,
        style: OverlayStyle
    ) {
        if (text.isBlank()) return
        if (mat.empty() || mat.cols() <= 0 || mat.rows() <= 0) return

        val safeBox = clampBoxToFrame(mat, box)

        // Compute text size
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(
            text,
            Imgproc.FONT_HERSHEY_SIMPLEX,
            style.fontScale,
            style.textThickness,
            baseline
        )

        val rawTextWidth = textSize.width.toInt()
        val textHeight = textSize.height.toInt()
        val base = baseline[0]

        val frameWidth = mat.cols()
        val frameHeight = mat.rows()

        // Label box dimensions with padding
        val labelWidthRaw = rawTextWidth + style.padding * 2
        val labelHeightRaw = textHeight + base + style.padding * 2

        val labelWidth = labelWidthRaw.coerceAtMost(frameWidth)
        val labelHeight = labelHeightRaw.coerceAtMost(frameHeight)

        // Position label horizontally near box left
        val preferredX = safeBox.x1.toInt()
        val maxLabelX = (frameWidth - labelWidth).coerceAtLeast(0)
        val labelX = preferredX.coerceIn(0, maxLabelX)

        val maxLabelY = (frameHeight - labelHeight).coerceAtLeast(0)
        val labelY = if (placeBelow) {
            val preferredBelowY = safeBox.y2.toInt() + style.margin
            if (preferredBelowY <= maxLabelY) {
                preferredBelowY
            } else {
                (safeBox.y2.toInt() - labelHeight).coerceIn(0, maxLabelY)
            }
        } else {
            val preferredAboveY = safeBox.y1.toInt() - labelHeight - style.margin
            if (preferredAboveY >= 0) {
                preferredAboveY.coerceIn(0, maxLabelY)
            } else {
                safeBox.y1.toInt().coerceIn(0, maxLabelY)
            }
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
        val textX = (rectLeft + style.padding).coerceAtMost((frameWidth - 1).coerceAtLeast(0))
        val textY = (rectBottom - style.padding - base).coerceIn(
            0,
            (frameHeight - 1).coerceAtLeast(0)
        )

        // Draw label text
        Imgproc.putText(
            mat,
            text,
            Point(textX.toDouble(), textY.toDouble()),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            style.fontScale,
            color,
            style.textThickness
        )
    }

    private data class LabelMetrics(
        val text: String,
        val textWidth: Int,
        val baseline: Int,
        val boxWidth: Int,
        val boxHeight: Int
    )

    /** Places user and emotion rows in one centered stack immediately above the box. */
    private fun drawCenteredInferenceLabels(
        mat: Mat,
        box: BoundingBox,
        labels: InferenceLabelFormatter.Labels,
        color: Scalar,
        style: OverlayStyle
    ) {
        val rows = listOfNotNull(labels.user, labels.emotion)
            .map { text ->
                val baseline = IntArray(1)
                val textSize = Imgproc.getTextSize(
                    text,
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    style.fontScale,
                    style.textThickness,
                    baseline
                )
                LabelMetrics(
                    text = text,
                    textWidth = textSize.width.toInt(),
                    baseline = baseline[0],
                    boxWidth = textSize.width.toInt() + style.padding * 2,
                    boxHeight = textSize.height.toInt() + baseline[0] + style.padding * 2
                )
            }
        if (rows.isEmpty()) return

        val frameWidth = mat.cols()
        val frameHeight = mat.rows()
        if (frameWidth <= 0 || frameHeight <= 0) return

        val safeBox = clampBoxToFrame(mat, box)
        val sharedWidth = rows.maxOf(LabelMetrics::boxWidth).coerceAtMost(frameWidth)
        val stackHeight = (
            rows.sumOf(LabelMetrics::boxHeight) + style.stackGap * (rows.size - 1)
        ).coerceAtMost(frameHeight)
        val boxCenterX = ((safeBox.x1 + safeBox.x2) / 2f).toInt()
        val stackLeft = (boxCenterX - sharedWidth / 2)
            .coerceIn(0, (frameWidth - sharedWidth).coerceAtLeast(0))
        val preferredTop = safeBox.y1.toInt() - style.margin - stackHeight
        var rowTop = preferredTop.coerceIn(0, (frameHeight - stackHeight).coerceAtLeast(0))

        rows.forEach { row ->
            val rowBottom = (rowTop + row.boxHeight).coerceAtMost(frameHeight)
            if (rowBottom <= rowTop) return@forEach

            Imgproc.rectangle(
                mat,
                Point(stackLeft.toDouble(), rowTop.toDouble()),
                Point((stackLeft + sharedWidth).toDouble(), rowBottom.toDouble()),
                Scalar(0.0, 0.0, 0.0),
                Imgproc.FILLED
            )

            val textX = (stackLeft + (sharedWidth - row.textWidth) / 2)
                .coerceIn(0, (frameWidth - 1).coerceAtLeast(0))
            val textY = (rowBottom - style.padding - row.baseline)
                .coerceIn(0, (frameHeight - 1).coerceAtLeast(0))
            Imgproc.putText(
                mat,
                row.text,
                Point(textX.toDouble(), textY.toDouble()),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                style.fontScale,
                color,
                style.textThickness
            )
            rowTop = rowBottom + style.stackGap
        }
    }

    // Draws bounding box + combined user/emotion label
    fun drawUserBoxAndLabel(mat: Mat, box: BoundingBox, classId: Int,
                            style: OverlayStyle = OverlayStyle.HANDHELD) {
        if (mat.empty() || mat.cols() <= 0 || mat.rows() <= 0) return

        val safeClassId = classId.coerceIn(0, 2)
        val safeBox = clampBoxToFrame(mat, box)

        // Skip invalid boxes
        if (safeBox.x2 <= safeBox.x1 || safeBox.y2 <= safeBox.y1) return
        val emotionDisplay = emotionDisplayForUser(safeClassId)

        // Select color based on current mode
        val borderColor = when (Settings.BoundingBox.colorMode) {
            Settings.BoundingBox.ColorMode.BY_USER -> emotionDisplay.color
            Settings.BoundingBox.ColorMode.BY_EMOTION -> emotionDisplay.color
        }

        // Draw bounding box
        Imgproc.rectangle(
            mat,
            Point(safeBox.x1.toDouble(), safeBox.y1.toDouble()),
            Point(safeBox.x2.toDouble(), safeBox.y2.toDouble()),
            borderColor,
            style.boxThickness
        )

        val labels = InferenceLabelFormatter.format(
            userNumber = safeClassId + 1,
            detectionConfidence = safeBox.confidence,
            emotion = emotionDisplay.emotion,
            emotionConfidencePct = emotionDisplay.confidence
        )

        drawCenteredInferenceLabels(mat, safeBox, labels, borderColor, style)
    }
}
