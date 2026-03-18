@file:Suppress("SameParameterValue")

package com.developer27.xemotion.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.developer27.xemotion.videoprocessing.CONTOUR.ContourVideoProcessing
import com.developer27.xemotion.videoprocessing.VideoDrawing.BoundingBox
import com.developer27.xemotion.videoprocessing.VideoDrawing.Traces.TracePatterns.KalmanBank
import com.developer27.xemotion.videoprocessing.VideoDrawing.VideoDrawingHelper
import com.developer27.xemotion.videoprocessing.YOLO.YOLOProcessing
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.tensorflow.lite.Interpreter
import java.util.LinkedList

// --------------------------------------------------
// Globals
// --------------------------------------------------
private var tfliteInterpreter: Interpreter? = null
val rawDataList = LinkedList<Point>()
val smoothDataList = LinkedList<Point>()

// --- Per-class traces (0=User_1, 1=User_2, 2=User_3) ---
val perClassRaw = Array(3) { LinkedList<Point>() }
val perClassSmooth = Array(3) { LinkedList<Point>() }

// Keep most-recent per-frame detection order (unique classIds 0..2)
private val lastDetectionOrder = LinkedList<Int>()

// Convenience: label map
private val USER_LABELS = arrayOf("User_1", "User_2", "User_3")

// Latest emotion & color per user (0..2)
private val perUserEmotion = Array(3) { "" }
private val perUserColor = Array(3) { Scalar(255.0, 255.0, 255.0) } // BGR
private val perUserConfidence = FloatArray(3) { Float.NaN } // percent 0..100

// --------------------------------------------------
// Settings
// --------------------------------------------------
object Settings {

    // Types of detections
    object DetectionMode {
        enum class Mode {
            CONTOUR,
            YOLO
        }
        var current: Mode = Mode.YOLO
        var enableYOLOinference = true
    }

    // Inference parameters
    object Inference {
        var confidenceThreshold: Float = 0.001f
        var iouThreshold: Float = 0.45f
    }

    // Trace settings
    object Trace {
        var enableRAWtrace = false
        var enableSPLINEtrace = true
        var splineStep = 0.01

        // Colors and thickness
        var originalLineColor = Scalar(173.0, 216.0, 230.0) // Light Blue
        var splineLineColor = Scalar(255.0, 203.0, 5.0)     // Maize
        var lineThickness = 100
    }

    object BoundingBox {
        // Enable bounding box or not
        var enableBoundingBox = true

        // used for contour bounding boxes:
        var boxColor = Scalar(255.0, 255.0, 255.0)
        var boxThickness = 10

        // max boxes per frame (1–3)
        var maxPerFrame: Int = 3

        // how to color the rectangle
        enum class ColorMode { BY_USER, BY_EMOTION }

        // color mode
        var colorMode: ColorMode = ColorMode.BY_USER
    }

    object Brightness {
        // Threshold and factor for brightness
        var factor = 2.0
        var threshold = 150.0
    }

    object ExportData {
        // Save the image or not
        var frameIMG = false

        // Do logging or not
        var enablePredictionLogging = false
    }

    object RollingShutter {
        // Set or update this from SettingsActivity
        var speedHz = 15f
    }
}

// --------------------------------------------------
// VideoProcessor
// --------------------------------------------------
class VideoProcessor(private val context: Context) {

    // Current user label (used in contour mode)
    var userLabel: String = "Contour"

    // Classification label (optional overlay text)
    var classificationLabel: String = ""

    // Return latest detection order (thread-safe)
    fun getDetectionOrder(): List<Int> = synchronized(lastDetectionOrder) {
        lastDetectionOrder.toList()
    }

    // Map classId (0..2) to user label string
    fun labelForClass(classId: Int): String =
        USER_LABELS[classId.coerceIn(0, 2)]

    // Get raw trace points per class
    fun getPerClassRaw(): Array<LinkedList<Point>> = perClassRaw

    // Get smoothed trace points per class
    fun getPerClassSmooth(): Array<LinkedList<Point>> = perClassSmooth

    // Get global smoothed trace (all users)
    fun getSmoothDataList(): List<Point> = smoothDataList

    // Get global raw trace (all users)
    fun getRawDataList(): List<Point> = rawDataList

    // Set emotion, color, and confidence for a user
    fun setPerUserEmotion(
        classId: Int,
        emotion: String,
        colorBGR: Scalar? = null,
        confidencePct: Float? = null
    ) {
        val id = classId.coerceIn(0, 2)
        perUserEmotion[id] = emotion
        perUserColor[id] = colorBGR ?: Scalar(255.0, 255.0, 255.0)
        confidencePct?.let { perUserConfidence[id] = it }
    }

    // YOLO processing pipeline (lazy initialization)
    private val yoloProcessing by lazy {
        YOLOProcessing(
            drawUserBoxAndLabel = { mat, box: BoundingBox, classId ->
                drawingHelper.drawUserBoxAndLabel(mat, box, classId)
            },
            perClassRaw = perClassRaw,
            lastDetectionOrder = lastDetectionOrder
        )
    }

    // Contour-based processing pipeline (lazy initialization)
    private val contourVideoProcessing by lazy {
        ContourVideoProcessing(
            drawTopLabel = { mat, box: BoundingBox, text, color ->
                drawingHelper.drawTopLabel(mat, box, text, color)
            },
            rawDataList = rawDataList,
            smoothDataList = smoothDataList
        )
    }

    // Helper for drawing bounding boxes, labels, etc.
    private val drawingHelper by lazy {
        VideoDrawingHelper(
            perUserEmotion = perUserEmotion,
            perUserColor = perUserColor,
            perUserConfidence = perUserConfidence
        )
    }

    // Initialize OpenCV and Kalman filter
    init {
        initOpenCV()
        KalmanBank.initKalmanFilter()
    }

    // Load OpenCV native library
    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
        } catch (e: UnsatisfiedLinkError) {
            Log.d("VideoProcessor", "OpenCV failed to load: ${e.message}", e)
        }
    }

    // Set and initialize TFLite interpreter
    fun setInterpreter(model: Interpreter) {

        // Thread-safe assignment of interpreter
        synchronized(this) {
            tfliteInterpreter = model
        }

        val interpreter = tfliteInterpreter

        // Debug: log input tensor information
        val inputCount = interpreter?.inputTensorCount ?: 0
        for (i in 0 until inputCount) {
            val t = interpreter?.getInputTensor(i)
            Log.d(
                "TFLITE_DEBUG",
                "input[$i] shape=${t?.shape()?.joinToString()} dtype=${t?.dataType()}"
            )
        }

        // Debug: log output tensor information
        val outputCount = interpreter?.outputTensorCount ?: 0
        for (i in 0 until outputCount) {
            val t = interpreter?.getOutputTensor(i)
            Log.d(
                "TFLITE_DEBUG",
                "output[$i] shape=${t?.shape()?.joinToString()} dtype=${t?.dataType()}"
            )
        }

        // Confirm interpreter setup
        Log.d("VideoProcessor", "TFLite Model set in VideoProcessor successfully!")
    }

    // Main async frame processing entry point
    fun processFrame(
        bitmap: Bitmap,
        callback: (Pair<Bitmap, Bitmap>?) -> Unit
    ) {
        // Debug: log current rolling shutter configuration
        Log.d("VideoProcessor", "Current Rolling Shutter Speed = ${Settings.RollingShutter.speedHz} Hz")

        // Run processing on background thread
        CoroutineScope(Dispatchers.Default).launch {
            val result: Pair<Bitmap, Bitmap>? = try {

                // Select detection pipeline (Contour vs YOLO)
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR -> processFrameInternalCONTOUR(bitmap)
                    Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
                }

            } catch (e: Exception) {
                // Handle processing errors safely
                Log.e("VideoProcessor", "Error processing frame: ${e.message}", e)
                null
            }

            // Return result to UI thread
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    // YOLO-based detection (runs on IO thread)
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Pair<Bitmap, Bitmap> =
        withContext(Dispatchers.IO) {
            yoloProcessing.processFrame(bitmap, tfliteInterpreter)
        }

    // Contour-based detection (no ML)
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        return contourVideoProcessing.processFrame(
            bitmap = bitmap,
            userLabel = userLabel,
            classificationLabel = classificationLabel
        )
    }

    // Clear all stored trace data
    fun reset() {
        rawDataList.clear()
        smoothDataList.clear()
    }
}