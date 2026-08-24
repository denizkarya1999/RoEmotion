@file:Suppress("SameParameterValue")

package com.developer27.xemotion.videoprocessing

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.OpenCVLoader
import com.developer27.xemotion.inference.YoloModelSession
import com.developer27.xemotion.videoprocessing.contour.ContourVideoProcessing
import com.developer27.xemotion.videoprocessing.drawing.BoundingBox
import com.developer27.xemotion.videoprocessing.drawing.VideoDrawingHelper
import com.developer27.xemotion.videoprocessing.drawing.traces.patterns.KalmanBank
import com.developer27.xemotion.videoprocessing.yolo.YoloProcessing
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Point
import org.opencv.core.Scalar

// Convenience: label map
private val USER_LABELS = arrayOf("User_1", "User_2", "User_3")

// --------------------------------------------------
// VideoProcessor
// --------------------------------------------------
class VideoProcessor {
    private val state = VideoProcessingState(USER_LABELS.size)
    private val traceStateLock = Any()
    private val processingScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var yoloModel: YoloModelSession? = null

    val isOpenCvReady: Boolean

    // Current user label (used in contour mode)
    @Volatile
    var userLabel: String = "Contour"

    // Classification label (optional overlay text)
    @Volatile
    var classificationLabel: String = ""

    // Map classId (0..2) to user label string
    fun labelForClass(classId: Int): String =
        USER_LABELS[classId.coerceIn(0, 2)]

    /** Atomically removes the YOLO traces accumulated since the previous inference tick. */
    fun drainDetectedTraceBatch(): Map<Int, List<Point>> = synchronized(traceStateLock) {
        state.detectionOrderSnapshot()
            .associateWith(state::drainPerUserTrace)
            .filterValues { points -> points.isNotEmpty() }
            .also { state.clearTraces() }
    }

    /**
     * Atomically removes contour traces only when all selected collection types can be rendered.
     * Keeping a partial trace prevents slow movement from being discarded on a timer tick.
     */
    fun drainReadyContourTraceBatch(
        types: Set<Settings.Trace.Type>
    ): Pair<List<Point>, List<Point>>? = synchronized(traceStateLock) {
        if (types.isEmpty()) return@synchronized null
        val raw = state.rawTraceSnapshot()
        val smooth = state.smoothTraceSnapshot()
        val ready = types.all { type ->
            val points = when (type) {
                Settings.Trace.Type.RAW,
                Settings.Trace.Type.RAW_CV -> raw
                Settings.Trace.Type.SPLINE,
                Settings.Trace.Type.SPLINE_CV -> smooth
            }
            points.size >= type.minimumPoints
        }
        if (!ready) return@synchronized null
        state.clearTraces()
        raw to smooth
    }

    /** Removes any remaining contour points, allowing valid partial final exports on Stop. */
    fun drainContourTraceBatch(): Pair<List<Point>, List<Point>> = synchronized(traceStateLock) {
        val batch = state.rawTraceSnapshot() to state.smoothTraceSnapshot()
        state.clearTraces()
        batch
    }

    // Set emotion, color, and confidence for a user
    fun setPerUserEmotion(
        classId: Int,
        emotion: String,
        colorBGR: Scalar? = null,
        confidencePct: Float? = null
    ) {
        state.setEmotion(classId, emotion, colorBGR, confidencePct)
    }

    // YOLO processing pipeline (lazy initialization)
    private val yoloProcessing by lazy {
        YoloProcessing(
            drawUserBoxAndLabel = { mat, box: BoundingBox, classId ->
                drawingHelper.drawUserBoxAndLabel(mat, box, classId)
            },
            recordTracePoint = state::appendPerUserTrace,
            updateDetectionOrder = state::recordDetections
        )
    }

    // Contour-based processing pipeline (lazy initialization)
    private val contourVideoProcessing by lazy {
        ContourVideoProcessing(
            drawTopLabel = { mat, box: BoundingBox, text, color ->
                drawingHelper.drawTopLabel(mat, box, text, color)
            },
            rawDataList = state.rawTrace,
            smoothDataList = state.smoothTrace
        )
    }

    // Helper for drawing bounding boxes, labels, etc.
    private val drawingHelper by lazy {
        VideoDrawingHelper(state::emotionDisplaySnapshot)
    }

    // Initialize OpenCV and Kalman filter
    init {
        isOpenCvReady = initOpenCV()
        KalmanBank.initKalmanFilter()
    }

    // Load the bundled OpenCV runtime using its version-aware loader.
    private fun initOpenCV(): Boolean = runCatching { OpenCVLoader.initLocal() }
        .onFailure { error -> Log.e(TAG, "OpenCV failed to load", error) }
        .getOrDefault(false)
        .also { loaded ->
            if (!loaded) Log.e(TAG, "OpenCV reported an unsuccessful load")
        }

    // Attach the loaded LiteRT compiled model.
    fun setYoloModel(model: YoloModelSession) {
        val previous = synchronized(this) {
            val old = yoloModel
            yoloModel = model
            old
        }
        previous?.let(::closeYoloModelSafely)

        Log.d(
            "VideoProcessor",
            "LiteRT model attached with ${model.accelerator}; " +
                "input=${model.inputShape.contentToString()} output=${model.outputShape.contentToString()}"
        )
    }

    fun clearYoloModel() {
        val previous = synchronized(this) {
            val old = yoloModel
            yoloModel = null
            old
        }
        previous?.let(::closeYoloModelSafely)
    }

    // Main async frame processing entry point
    fun processFrame(
        bitmap: Bitmap,
        callback: (Bitmap?) -> Unit
    ) {
        if (!isOpenCvReady) {
            Log.e(TAG, "Frame processing skipped because OpenCV is unavailable")
            if (!bitmap.isRecycled) bitmap.recycle()
            callback(null)
            return
        }

        // Debug: log current rolling shutter configuration
        Log.d("VideoProcessor", "Current Rolling Shutter Speed = ${Settings.RollingShutter.speedHz} Hz")

        // Run processing on background thread
        processingScope.launch {
            var result: Bitmap? = null
            var delivered = false
            try {
                result = when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR -> processFrameInternalCONTOUR(bitmap)
                    Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
                }
                withContext(Dispatchers.Main) {
                    delivered = true
                    callback(result)
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: LinkageError) {
                Log.e(TAG, "Native frame processing failed", error)
                withContext(Dispatchers.Main) {
                    delivered = true
                    callback(null)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    delivered = true
                    callback(null)
                }
            } finally {
                if (!bitmap.isRecycled) bitmap.recycle()
                if (!delivered) result?.takeUnless(Bitmap::isRecycled)?.recycle()
            }
        }
    }

    // YOLO-based detection (runs on IO thread)
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Bitmap =
        withContext(Dispatchers.IO) {
            synchronized(traceStateLock) {
                synchronized(this@VideoProcessor) {
                    val model = yoloModel
                    if (model == null) {
                        yoloProcessing.processFrame(bitmap, null)
                    } else {
                        synchronized(model) {
                            yoloProcessing.processFrame(bitmap, model)
                        }
                    }
                }
            }
        }

    // Contour-based detection (no ML)
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Bitmap {
        return synchronized(traceStateLock) {
            contourVideoProcessing.processFrame(
                bitmap = bitmap,
                userLabel = userLabel,
                classificationLabel = classificationLabel
            )
        }
    }

    // Clear all stored trace data
    fun reset() {
        synchronized(traceStateLock) { state.clearTraces() }
    }

    fun clearEmotionState() {
        state.clearEmotions()
    }

    fun close() {
        processingScope.cancel()
        val model = synchronized(this) {
            val current = yoloModel
            yoloModel = null
            current
        }
        model?.let(::closeYoloModelSafely)
    }

    private fun closeYoloModelSafely(model: YoloModelSession) {
        runCatching { synchronized(model) { model.close() } }
            .onFailure { error -> Log.e(TAG, "Unable to close YOLO model", error) }
    }

    private companion object {
        const val TAG = "VideoProcessor"
    }
}
