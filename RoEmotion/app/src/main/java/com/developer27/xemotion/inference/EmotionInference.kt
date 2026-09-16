package com.developer27.xemotion.inference

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.developer27.xemotion.storage.MediaStoreRepository
import com.developer27.xemotion.videoprocessing.LineProcessing
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoProcessor
import com.developer27.xemotion.videoprocessing.drawing.traces.patterns.KalmanBank
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.opencv.core.Point
import org.opencv.core.Scalar
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

/** Owns emotion inference, prediction state, and sequence buffering. */
class EmotionInference(
    context: Context,
    private val videoProcessor: VideoProcessor?
) {
    private val storage = MediaStoreRepository(context.applicationContext)
    private val modelLoader = PyTorchModelLoader(context.applicationContext)
    private val modelScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val processingScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val lineProcessing = LineProcessing(storage)
    private val perUserBuffers = Array(3) { ArrayDeque<Bitmap>() }
    private val kalmanManagedUsers = BooleanArray(perUserBuffers.size)
    private val traceBuffer = InferenceTraceBuffer(
        userCount = perUserBuffers.size,
        capacity = TRACE_POINT_CAPACITY
    )
    private var modelLoadJob: Job? = null
    private var sessionJob: Job? = null
    private var stopJob: Job? = null
    private var sessionMode = Settings.OperatingMode.Mode.INFERENCE
    private var activeSessionContext: SessionContext? = null
    private val classifierLock = Any()
    private val processingMutex = Mutex()
    private val modelLoadMutex = Mutex()
    private val modelGeneration = AtomicLong()
    private var requestedModelId: String? = null

    @Volatile
    private var loadedModelId: String? = null

    @Volatile
    private var closed = false

    @Volatile
    private var modelsLoaded = false

    private var emotionClassifier: PyTorchClassifier? = null

    /** Reloads when the selection changes, discarding obsolete loads and prediction state. */
    fun loadModels(onResult: (Result<Unit>) -> Unit = {}) {
        if (closed) return
        val selected = modelLoader.selectedModel()
        if (modelsLoaded && loadedModelId == selected.id) {
            onResult(Result.success(Unit))
            return
        }
        if (modelLoadJob?.isActive == true && requestedModelId == selected.id) return

        val generation = modelGeneration.incrementAndGet()
        requestedModelId = selected.id
        modelsLoaded = false
        loadedModelId = null
        clearAllPredictions()
        modelLoadJob?.cancel()
        modelLoadJob = modelScope.launch {
            // Cancellation cannot interrupt native loading. Keep ownership until cleanup finishes.
            modelLoadMutex.withLock {
                processingMutex.withLock {
                    if (closed || generation != modelGeneration.get()) return@launch
                    discardEmotionModel()
                }
                val loaded = try {
                    modelLoader.loadModels(selected)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Throwable) {
                    Log.e(TAG, "Unable to load inference models", error)
                    withContext(Dispatchers.Main) {
                        if (!closed && generation == modelGeneration.get()) {
                            requestedModelId = null
                            onResult(Result.failure(error))
                        }
                    }
                    return@launch
                }

                var attached = false
                try {
                    processingMutex.withLock {
                        withContext(Dispatchers.Main) {
                            if (closed || generation != modelGeneration.get() ||
                                selected.id != modelLoader.selectedModel().id
                            ) return@withContext
                            clearBuffers()
                            videoProcessor?.reset()
                            clearAllPredictions()
                            videoProcessor?.setYoloModel(loaded.yoloModel)
                            synchronized(classifierLock) {
                                emotionClassifier = loaded.emotionClassifier
                            }
                            loadedModelId = selected.id
                            modelsLoaded = true
                            attached = true
                            Log.i(TAG, "Inference models are ready: ${selected.displayName}")
                            onResult(Result.success(Unit))
                        }
                    }
                } finally {
                    if (!attached) loaded.close()
                }
            }
        }
    }

    fun areModelsLoaded(): Boolean =
        modelsLoaded && loadedModelId == modelLoader.selectedModel().id

    /** Releases inference-only resources when Data Collection Mode is selected. */
    fun unloadModels() {
        val generation = modelGeneration.incrementAndGet()
        modelLoadJob?.cancel()
        requestedModelId = null
        modelsLoaded = false
        loadedModelId = null
        clearAllPredictions()
        modelLoadJob = modelScope.launch {
            modelLoadMutex.withLock {
                processingMutex.withLock {
                    if (generation != modelGeneration.get()) return@launch
                    discardEmotionModel()
                    videoProcessor?.clearYoloModel()
                }
            }
        }
    }

    /** Must be called while processingMutex is held. */
    private fun discardEmotionModel() {
        synchronized(classifierLock) {
            runCatching { emotionClassifier?.close() }
                .onFailure { error -> Log.e(TAG, "Unable to close classifier", error) }
            emotionClassifier = null
        }
        clearBuffers()
        videoProcessor?.reset()
        clearAllPredictions()
    }

    /** Resets state and starts the mode-specific periodic inference loop. */
    fun startSession(
        collectionUserName: String? = null,
        collectionEmotion: String? = null
    ) {
        check(!closed) { "EmotionInference has been closed." }
        val previousSession = sessionJob
        previousSession?.cancel()
        sessionMode = Settings.OperatingMode.current
        val sessionContext = SessionContext(collectionUserName, collectionEmotion)
        activeSessionContext = sessionContext

        val intervalMillis = when (sessionMode) {
            Settings.OperatingMode.Mode.DATA_COLLECTION -> CONTOUR_INTERVAL_MILLIS
            Settings.OperatingMode.Mode.INFERENCE -> YOLO_INTERVAL_MILLIS
        }

        val mode = sessionMode
        sessionJob = processingScope.launch {
            previousSession?.cancelAndJoin()
            processingMutex.withLock { initializeSession(sessionContext) }
            while (isActive) {
                delay(intervalMillis)
                try {
                    processingMutex.withLock { processExportTick(mode) }
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
        val sessionContext = activeSessionContext
        sessionJob = null
        activeSessionContext = null
        stopJob?.cancel()
        stopJob = processingScope.launch {
            activeSession?.cancelAndJoin()
            processingMutex.withLock {
                // Stop can arrive before the worker has initialized a newly started session.
                sessionContext?.let(::initializeSession)
                finishProcessing(mode)
                clearBuffers()
                videoProcessor?.classificationLabel = ""
            }
            val savedCount = lineProcessing.savedTraceCount
                .takeIf { mode == Settings.OperatingMode.Mode.DATA_COLLECTION }
            withContext(Dispatchers.Main) {
                if (!closed) onStopped(savedCount)
            }
        }
    }

    /** Must be called while processingMutex is held. */
    private fun initializeSession(session: SessionContext) {
        if (session.initialized) return
        lineProcessing.beginSession(session.collectionUserName, session.collectionEmotion)
        clearBuffers()
        videoProcessor?.reset()
        clearAllPredictions()
        session.initialized = true
    }

    private fun clearUserFrames(classId: Int) {
        val buffer = perUserBuffers[classId]
        while (buffer.isNotEmpty()) buffer.removeFirst().recycle()
        if (kalmanManagedUsers[classId]) {
            KalmanBank.reset(classId)
            kalmanManagedUsers[classId] = false
        }
    }

    private fun clearBuffers() {
        perUserBuffers.indices.forEach(::clearUserFrames)
        traceBuffer.clear()
    }

    private fun prepareInferenceTrace(
        classId: Int,
        newPoints: List<Point>,
        traceType: Settings.Trace.Type
    ): Bitmap? {
        val model = emotionClassifier?.modelSpec ?: return null
        if (!model.isLegacy && traceBuffer.recordObservation(
                classId,
                SystemClock.elapsedRealtime(),
                CURRENT_MODEL_MAX_GAP_MILLIS
            )
        ) {
            clearUserFrames(classId)
            videoProcessor?.let { processor ->
                processor.setPerUserEmotion(classId, "", DEFAULT_COLOR, Float.NaN)
                if (processor.userLabel == processor.labelForClass(classId)) {
                    processor.userLabel = ""
                    processor.classificationLabel = ""
                }
            }
        }
        val useKalman = model.traceVariant == EmotionTraceVariant.KALMAN_CV
        if (useKalman && !kalmanManagedUsers[classId]) {
            KalmanBank.reset(classId)
            kalmanManagedUsers[classId] = true
        }
        val points = traceBuffer.appendAndSnapshot(
            classId = classId,
            newPoints = newPoints,
            minimumPoints = traceType.minimumPoints
        ) { point ->
            if (useKalman) {
                val (x, y) = KalmanBank.applyKalmanFilter(point, classId)
                Point(x, y)
            } else point
        } ?: return null
        return lineProcessing.exportProcessedTraceForClass(points, traceType)
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
        if (!areModelsLoaded()) return
        val sequenceLength = emotionClassifier?.modelSpec?.sequenceLength ?: return
        val traceType = Settings.Trace.inferenceType
        val detectedTraces = processor.drainDetectedTraceBatch()
        for ((classId, newPoints) in detectedTraces) {
            if (classId !in perUserBuffers.indices) continue

            val trace = prepareInferenceTrace(classId, newPoints, traceType) ?: continue

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
                    if (!areModelsLoaded()) return
                    val sequenceLength = emotionClassifier?.modelSpec?.sequenceLength ?: return
                    val traceType = Settings.Trace.inferenceType
                    for ((classId, newPoints) in processor.drainDetectedTraceBatch()) {
                        if (classId !in perUserBuffers.indices) continue
                        val trace = prepareInferenceTrace(classId, newPoints, traceType) ?: continue
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
        // A preference change may happen while native inference is completing.
        if (!areModelsLoaded()) return
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
        if (!areModelsLoaded()) return null
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
        label.contains("Angry", true) || label.equals("Anger", true) -> Scalar(255.0, 102.0, 102.0)
        label.contains("Anxi", true) -> Scalar(255.0, 255.0, 153.0)
        label.contains("Disgust", true) -> Scalar(153.0, 255.0, 153.0)
        label.contains("Excit", true) -> Scalar(255.0, 204.0, 153.0)
        label.contains("Sad", true) -> Scalar(153.0, 204.0, 255.0)
        else -> DEFAULT_COLOR
    }

    fun close() {
        if (closed) return
        closed = true
        modelGeneration.incrementAndGet()
        val activeSession = sessionJob
        val activeStop = stopJob
        sessionJob = null
        activeSessionContext = null
        stopJob = null
        modelLoadJob?.cancel()
        modelLoadJob = null
        modelScope.cancel()
        modelsLoaded = false
        loadedModelId = null
        processingScope.launch {
            activeSession?.cancelAndJoin()
            activeStop?.cancelAndJoin()
            processingMutex.withLock { discardEmotionModel() }
            processingScope.cancel()
        }
    }

    private data class SessionContext(
        val collectionUserName: String?,
        val collectionEmotion: String?,
        var initialized: Boolean = false
    )

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
        const val CURRENT_MODEL_MAX_GAP_MILLIS = 10_000L
        const val PREDICTION_LOG_FILE = "RoEmotion_Predictions_Log.txt"
        val DEFAULT_COLOR = Scalar(255.0, 255.0, 255.0)
    }
}
