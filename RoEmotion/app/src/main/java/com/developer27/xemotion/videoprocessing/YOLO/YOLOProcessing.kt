package com.developer27.xemotion.videoprocessing.YOLO

import android.graphics.Bitmap
import android.util.Log
import com.developer27.xemotion.inference.YOLO_LED_Detection
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoDrawing.BoundingBox
import com.developer27.xemotion.videoprocessing.VideoDrawing.Traces.TraceRenderer
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Point
import org.tensorflow.lite.Interpreter
import java.util.LinkedList

class YOLOProcessing(
    private val drawUserBoxAndLabel: (Mat, BoundingBox, Int) -> Unit,
    private val perClassRaw: Array<LinkedList<Point>>,
    private val lastDetectionOrder: LinkedList<Int>
) {

    companion object {
        private const val TAG = "YOLOProcessing"
        private const val INPUT_W = 640
        private const val INPUT_H = 640
        private const val TRACE_CAP = 300
    }

    // Main YOLO frame processing pipeline
    fun processFrame(
        bitmap: Bitmap,
        interpreter: Interpreter?
    ): Pair<Bitmap, Bitmap> {

        // Resize + pad input to match model input size (letterboxing)
        val meta = YOLO_LED_Detection.createLetterboxedBitmap(
            srcBitmap = bitmap,
            targetWidth = INPUT_W,
            targetHeight = INPUT_H
        )
        val letterboxed = meta.inputBitmap

        // convert original to Mat for drawing
        val m = Mat()
        Utils.bitmapToMat(bitmap, m)

        // Run YOLO inference if enabled and interpreter is available
        if (Settings.DetectionMode.enableYOLOinference && interpreter != null) {

            // Get model output tensor shape
            val shape = interpreter.getOutputTensor(0).shape()

            // Convert input image to normalized tensor (NHWC format)
            val inputBuffer = YOLO_LED_Detection.bitmapToNormalizedTensorNHWC(letterboxed)

            // Allocate output buffer dynamically based on model shape
            val out = when {
                shape.size == 3 -> {
                    val batch = shape[0]
                    val dim1 = shape[1]
                    val dim2 = shape[2]
                    Array(batch) { Array(dim1) { FloatArray(dim2) } }
                }
                else -> {
                    // Handle unsupported model output
                    Log.e(TAG, "Unsupported output tensor shape: ${shape.contentToString()}")
                    val debugOverlay = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
                    Utils.matToBitmap(m, debugOverlay)
                    m.release()
                    return debugOverlay to letterboxed
                }
            }

            // Run TFLite inference
            interpreter.run(inputBuffer, out)

            // Parse detections + apply confidence filtering + NMS
            val detsAll = YOLO_LED_Detection.parseTFLite(out)

            // Keep one detection per class and limit total boxes
            val perClassDets = YOLO_LED_Detection
                .pickOnePerClass(detsAll, nClasses = 3)
                .take(Settings.BoundingBox.maxPerFrame)

            // Map detections back to original image coordinates
            val rescaled = YOLO_LED_Detection.rescaleDetections(
                detections = perClassDets,
                originalWidth = bitmap.width,
                originalHeight = bitmap.height,
                padOffsets = meta.padLeft to meta.padTop,
                modelInputWidth = INPUT_W,
                modelInputHeight = INPUT_H,
                scale = meta.scale
            )

            // Update last detection order (used for tracking / labeling consistency)
            synchronized(lastDetectionOrder) {
                lastDetectionOrder.clear()
                lastDetectionOrder.addAll(
                    perClassDets.map { it.classId.coerceIn(0, 2) }
                )
            }

            // Draw detections and update motion traces
            rescaled.forEachIndexed { idx, (box, centerPoint) ->
                val det = perClassDets[idx]
                val classId = det.classId.coerceIn(0, YOLO_LED_Detection.userLabels.lastIndex)

                // Store trajectory points (for motion/gesture analysis)
                perClassRaw[classId].add(centerPoint)
                if (perClassRaw[classId].size > TRACE_CAP) {
                    perClassRaw[classId].removeFirst()
                }

                // Draw bounding box if enabled
                if (Settings.BoundingBox.enableBoundingBox &&
                    box.x2 > box.x1 &&
                    box.y2 > box.y1
                ) {
                    drawUserBoxAndLabel(m, box, classId)
                }

                // Draw trajectory path inside bounding box
                TraceRenderer.drawPathInBox(m, box, perClassRaw[classId])
            }
        }

        // Convert processed Mat back to Bitmap
        val debugOverlay = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(m, debugOverlay)
        m.release()

        return debugOverlay to letterboxed
    }
}