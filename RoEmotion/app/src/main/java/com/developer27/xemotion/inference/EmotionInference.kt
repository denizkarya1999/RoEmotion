package com.developer27.xemotion.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.developer27.xemotion.storage.MediaStoreRepository
import com.developer27.xemotion.videoprocessing.LineProcessing
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoProcessor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Point
import org.opencv.core.Scalar
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale

/** Owns emotion inference, prediction state, and sequence buffering. */
class EmotionInference(
    context: Context,
    private val videoProcessor: VideoProcessor?,
    private val sequenceLength: Int = 5
) {
    private val storage = MediaStoreRepository(context.applicationContext)
    private val modelLoader = PyTorchModelLoader(context.applicationContext)
    private val modelScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val processingScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val lineProcessing = LineProcessing(storage)
    private val perUserBuffers = Array(3) { ArrayDeque<Bitmap>(sequenceLength) }
    private val traceBuffer = InferenceTraceBuffer(
        userCount = perUserBuffers.size,
        capacity = TRACE_POINT_CAPACITY
    )
    private var modelLoadJob: Job? = null
    private var sessionJob: Job? = null
    private var stopJob: Job? = null
    private var sessionMode = Settings.OperatingMode.Mode.INFERENCE
    private val classifierLock = Any()

    @Volatile
    private var closed = false

    @Volatile
    private var modelsLoaded = false

    private var emotionClassifier: PyTorchClassifier? = null

    /** Loads both bundled models off the UI thread and attaches them atomically. */
    fun loadModels(onResult: (Result<Unit>) -> Unit = {}) {
        if (modelsLoaded) {
            onResult(Result.success(Unit))
            return
        }
        if (modelLoadJob?.isActive == true) return

        modelLoadJob?.cancel()
        modelLoadJob = modelScope.launch {
            val loaded = try {
                modelLoader.loadModels()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(TAG, "Unable to load inference models", error)
                withContext(Dispatchers.Main) { onResult(Result.failure(error)) }
                return@launch
            }

            var attached = false
            try {
                withContext(Dispatchers.Main) {
                    videoProcessor?.setYoloModel(loaded.yoloModel)
                    synchronized(classifierLock) {
                        runCatching { emotionClassifier?.close() }
                            .onFailure { error -> Log.e(TAG, "Unable to close old classifier", error) }
                        emotionClassifier = loaded.emotionClassifier
                    }
                    modelsLoaded = true
                    attached = true
                    Log.i(TAG, "YOLO and emotion-classification models are ready")
                    onResult(Result.success(Unit))
                }
            } finally {
                if (!attached) loaded.close()
            }
        }
    }

    fun areModelsLoaded(): Boolean = modelsLoaded

    /** Releases inference-only resources when Data Collection Mode is selected. */
    fun unloadModels() {
        modelLoadJob?.cancel()
        modelLoadJob = null
        modelsLoaded = false
        synchronized(classifierLock) {
            runCatching { emotionClassifier?.close() }
                .onFailure { error -> Log.e(TAG, "Unable to close classifier", error) }
            emotionClassifier = null
        }
        videoProcessor?.clearYoloModel()
    }

    /** Resets state and starts the mode-specific periodic inference loop. */
    fun startSession(
        collectionUserName: String? = null,
        collectionEmotion: String? = null
    ) {
        check(!closed) { "EmotionInference has been closed." }
        sessionJob?.cancel()
        lineProcessing.beginSession(collectionUserName, collectionEmotion)
        clearBuffers()
        videoProcessor?.reset()
        clearAllPredictions()
        sessionMode = Settings.OperatingMode.current

        val intervalMillis = when (sessionMode) {
            Settings.OperatingMode.Mode.DATA_COLLECTION -> CONTOUR_INTERVAL_MILLIS
            Settings.OperatingMode.Mode.INFERENCE -> YOLO_INTERVAL_MILLIS
        }

        val mode = sessionMode
        sessionJob = processingScope.launch {
            while (isActive) {
                delay(intervalMillis)
                try {
                    processExportTick(mode)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    Log.e(TAG, "Periodic processing failed", error)
                }
            }
        }
    }

    /** Stops processing asynchronously after any active inference/export operation finishes. */
    fun stopSession(onStopped: (Int?) -> Unit) {
        val activeSession = sessionJob
        val mode = sessionMode
        sessionJob = null
        stopJob?.cancel()
        stopJob = processingScope.launch {
            activeSession?.cancelAndJoin()
            finishProcessing(mode)
            clearBuffers()
            videoProcessor?.classificationLabel = ""
            val savedCount = lineProcessing.savedTraceCount
                .takeIf { mode == Settings.OperatingMode.Mode.DATA_COLLECTION }
            withContext(Dispatchers.Main) {
                if (!closed) onStopped(savedCount)
            }
        }
    }

    private fun clearBuffers() {
        perUserBuffers.forEach { buffer ->
            while (buffer.isNotEmpty()) buffer.removeFirst().recycle()
        }
        traceBuffer.clear()
    }

    fun clearAllPredictions() {
        videoProcessor?.classificationLabel = ""
        videoProcessor?.userLabel = ""
        videoProcessor?.clearEmotionState()
    }

    private fun processExportTick(mode: Settings.OperatingMode.Mode) {
        val processor = videoProcessor ?: return
        when (mode) {
            Settings.OperatingMode.Mode.DATA_COLLECTION -> processDataCollectionTick(processor)
            Settings.OperatingMode.Mode.INFERENCE -> processYoloInferenceTick(processor)
        }
    }

    private fun processYoloInferenceTick(processor: VideoProcessor) {
        val traceType = Settings.Trace.inferenceType
        val detectedTraces = processor.drainDetectedTraceBatch()
        for ((classId, newPoints) in detectedTraces) {
            if (classId !in perUserBuffers.indices) continue

            val points = traceBuffer.appendAndSnapshot(
                classId = classId,
                newPoints = newPoints,
                minimumPoints = traceType.minimumPoints
            ) ?: continue
            val trace = lineProcessing.exportProcessedTraceForClass(
                points,
                traceType
            ) ?: continue

            val buffer = perUserBuffers[classId]
            buffer.addLast(trace)
            if (buffer.size > sequenceLength) buffer.removeFirst().recycle()
            if (buffer.size == sequenceLength) {
                val frames = buffer.toList()
                try {
                    Log.d(TAG, "Running emotion inference for User_${classId + 1}")
                    runPerUserInferenceAndLabel(classId, frames)
                } finally {
                    frames.forEach(Bitmap::recycle)
                    buffer.clear()
                }
            }
        }
    }

    private fun processDataCollectionTick(processor: VideoProcessor) {
        val types = Settings.Trace.collectionTypes
        val (rawPoints, smoothPoints) = processor.drainReadyContourTraceBatch(types) ?: return
        exportSelectedDataCollectionTraces(rawPoints, smoothPoints, types)
    }

    private fun exportSelectedDataCollectionTraces(
        rawPoints: List<Point>,
        smoothPoints: List<Point>,
        types: Set<Settings.Trace.Type>
    ) {
        val traces = lineProcessing.exportSelectedContourTraces(
            rawPoints,
            smoothPoints,
            types
        )
        try {
            lineProcessing.saveContourTraces(traces)
        } finally {
            traces.values.forEach(Bitmap::recycle)
        }
    }

    private fun finishProcessing(mode: Settings.OperatingMode.Mode) {
        try {
            val processor = videoProcessor ?: return
            when (mode) {
                Settings.OperatingMode.Mode.DATA_COLLECTION -> {
                    val (rawPoints, smoothPoints) = processor.drainContourTraceBatch()
                    exportSelectedDataCollectionTraces(
                        rawPoints,
                        smoothPoints,
                        Settings.Trace.collectionTypes
                    )
                }
                Settings.OperatingMode.Mode.INFERENCE -> {
                    val traceType = Settings.Trace.inferenceType
                    for ((classId, newPoints) in processor.drainDetectedTraceBatch()) {
                        if (classId !in perUserBuffers.indices) continue
                        val points = traceBuffer.appendAndSnapshot(
                            classId = classId,
                            newPoints = newPoints,
                            minimumPoints = traceType.minimumPoints
                        ) ?: continue
                        val trace = lineProcessing.exportProcessedTraceForClass(
                            points,
                            traceType
                        ) ?: continue
                        val buffer = perUserBuffers[classId]
                        buffer.addLast(trace)
                        while (buffer.size > sequenceLength) buffer.removeFirst().recycle()
                        if (buffer.size >= sequenceLength) {
                            val frames = buffer.toList()
                            try {
                                runPerUserInferenceAndLabel(classId, frames)
                            } finally {
                                frames.forEach(Bitmap::recycle)
                                buffer.clear()
                            }
                        }
                    }
                }
            }
        } catch (exception: Exception) {
            Log.e(TAG, "Error completing emotion inference", exception)
        }
    }

    private fun appendPredictionToLog(prediction: String) {
        if (!Settings.ExportData.enablePredictionLogging) return
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
        val line = "$timestamp => $prediction"
        storage.appendDocumentLine(PREDICTION_LOG_FILE, line)
    }

    private fun runPerUserInferenceAndLabel(classId: Int, frames: List<Bitmap>) {
        val prediction = classify(frames) ?: return
        val userName = videoProcessor?.labelForClass(classId) ?: "User_${classId + 1}"
        val text = String.format(
            Locale.US,
            "%s | %s (%.2f%%)",
            userName,
            prediction.label,
            prediction.confidencePct
        )
        videoProcessor?.setPerUserEmotion(
            classId,
            prediction.label,
            prediction.color,
            prediction.confidencePct
        )
        videoProcessor?.userLabel = userName
        videoProcessor?.classificationLabel = text
        Log.i(TAG, "Prediction: $text")
        appendPredictionToLog(text)
    }

    private fun classify(frames: List<Bitmap>): EmotionPrediction? {
        val (label, probabilities) = synchronized(classifierLock) {
            val classifier = emotionClassifier ?: return null
            classifier.classifySequence(frames).also { (_, values) ->
                classifier.logProbabilities(values)
            }
        }
        val bestIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidencePct = probabilities.getOrElse(bestIndex) { 0f } * 100f
        return EmotionPrediction(label, confidencePct, colorForEmotion(label))
    }

    private fun colorForEmotion(label: String): Scalar = when {
        label.contains("Angry", true) -> Scalar(255.0, 102.0, 102.0)
        label.contains("Anxi", true) -> Scalar(255.0, 255.0, 153.0)
        label.contains("Disgust", true) -> Scalar(153.0, 255.0, 153.0)
        label.contains("Excit", true) -> Scalar(255.0, 204.0, 153.0)
        label.contains("Sad", true) -> Scalar(153.0, 204.0, 255.0)
        else -> DEFAULT_COLOR
    }

    fun close() {
        if (closed) return
        closed = true
        val activeSession = sessionJob
        val activeStop = stopJob
        sessionJob = null
        stopJob = null
        modelLoadJob?.cancel()
        modelLoadJob = null
        modelScope.cancel()
        modelsLoaded = false
        processingScope.launch {
            activeSession?.cancelAndJoin()
            activeStop?.cancelAndJoin()
            clearBuffers()
            synchronized(classifierLock) {
                runCatching { emotionClassifier?.close() }
                    .onFailure { error -> Log.e(TAG, "Unable to close classifier", error) }
                emotionClassifier = null
            }
            processingScope.cancel()
        }
    }

    private data class EmotionPrediction(
        val label: String,
        val confidencePct: Float,
        val color: Scalar
    )

    private companion object {
        const val TAG = "EmotionInference"
        const val CONTOUR_INTERVAL_MILLIS = 700L
        const val YOLO_INTERVAL_MILLIS = 800L
        const val TRACE_POINT_CAPACITY = 300
        const val PREDICTION_LOG_FILE = "RoEmotion_Predictions_Log.txt"
        val DEFAULT_COLOR = Scalar(255.0, 255.0, 255.0)
    }
}
