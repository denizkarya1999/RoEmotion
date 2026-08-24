package com.developer27.xemotion.videoprocessing.yolo

import android.graphics.Bitmap
import android.util.Log
import com.developer27.xemotion.inference.YoloDet
import com.developer27.xemotion.inference.YoloLedDetection
import com.developer27.xemotion.inference.YoloModelSession
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.drawing.BoundingBox
import com.developer27.xemotion.videoprocessing.drawing.traces.TraceRenderer
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Point

class YoloProcessing(
    private val drawUserBoxAndLabel: (Mat, BoundingBox, Int) -> Unit,
    private val recordTracePoint: (Int, Point, Int) -> List<Point>,
    private val updateDetectionOrder: (List<Int>) -> Unit
) {

    companion object {
        private const val TAG = "YoloProcessing"
        private const val INPUT_W = 640
        private const val INPUT_H = 640
        private const val TRACE_CAP = 300
    }

    // Main YOLO frame processing pipeline
    fun processFrame(
        bitmap: Bitmap,
        model: YoloModelSession?
    ): Bitmap {
        val outputMat = Mat()
        return try {
            Utils.bitmapToMat(bitmap, outputMat)
            if (Settings.DetectionMode.enableYOLOinference && model != null) {
                detectAndDraw(bitmap, model, outputMat)
            }

            Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888).also {
                Utils.matToBitmap(outputMat, it)
            }
        } finally {
            outputMat.release()
        }
    }

    private fun detectAndDraw(bitmap: Bitmap, model: YoloModelSession, outputMat: Mat) {
        val meta = YoloLedDetection.createLetterboxedBitmap(
            srcBitmap = bitmap,
            targetWidth = INPUT_W,
            targetHeight = INPUT_H
        )
        val letterboxed = meta.inputBitmap

        try {
            val shape = model.outputShape
            if (shape.size != 3) {
                Log.e(TAG, "Unsupported output tensor shape: ${shape.contentToString()}")
                return
            }

            val input = YoloLedDetection.bitmapToNormalizedTensorNHWC(letterboxed)
            val output = model.run(input)

            val detections = YoloLedDetection
                .pickOnePerClass(YoloLedDetection.parseTFLite(output, shape), nClasses = 3)
                .sortedByDescending(YoloDet::objConf)
                .take(Settings.BoundingBox.maxPerFrame)
            val rescaled = YoloLedDetection.rescaleDetections(
                detections = detections,
                originalWidth = bitmap.width,
                originalHeight = bitmap.height,
                padOffsets = meta.padLeft to meta.padTop,
                modelInputWidth = INPUT_W,
                modelInputHeight = INPUT_H,
                scale = meta.scale
            )

            updateDetectionOrder(detections.map { it.classId.coerceIn(0, 2) })
            rescaled.forEachIndexed { index, (box, centerPoint) ->
                val classId = detections[index].classId.coerceIn(
                    0,
                    YoloLedDetection.userLabels.lastIndex
                )
                val trace = recordTracePoint(classId, centerPoint, TRACE_CAP)

                if (Settings.BoundingBox.enableBoundingBox && box.x2 > box.x1 && box.y2 > box.y1) {
                    drawUserBoxAndLabel(outputMat, box, classId)
                }
                TraceRenderer.drawPathInBox(outputMat, box, trace)
            }
        } finally {
            if (!letterboxed.isRecycled) letterboxed.recycle()
        }
    }
}
