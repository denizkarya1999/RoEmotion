package com.developer27.xemotion.inference

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import org.pytorch.Module
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean

/** Owns loading of all bundled inference models. Call its methods from a worker thread. */
class PyTorchModelLoader(context: Context) {
    private val appContext = context.applicationContext

    fun loadModels(): LoadedInferenceModels {
        val yoloModel = loadYoloModel()
        return try {
            LoadedInferenceModels(yoloModel, loadEmotionClassifier())
        } catch (error: Throwable) {
            yoloModel.close()
            throw error
        }
    }

    fun loadYoloModel(): YoloModelSession {
        return createYoloModel(CompiledModel.Options(Accelerator.GPU), "GPU")
            ?: checkNotNull(createYoloModel(CompiledModel.Options.CPU, "CPU"))
    }

    fun loadEmotionClassifier(): PyTorchClassifier {
        val modelFile = copyAssetToInternalStorage(EMOTION_MODEL_ASSET)
        return PyTorchClassifier(Module.load(modelFile.absolutePath))
    }

    private fun createYoloModel(
        options: CompiledModel.Options,
        accelerator: String
    ): YoloModelSession? = runCatching {
        val compiledModel = CompiledModel.create(
            appContext.assets,
            YOLO_MODEL_ASSET,
            options
        )
        YoloModelSession(compiledModel, accelerator)
    }.onSuccess {
        Log.i(TAG, "Loaded YOLO model with LiteRT $accelerator")
    }.onFailure { error ->
        Log.w(TAG, "LiteRT $accelerator initialization failed", error)
    }.getOrNull()

    private fun copyAssetToInternalStorage(assetName: String): File {
        synchronized(ASSET_COPY_LOCK) {
            return copyAssetToInternalStorageLocked(assetName)
        }
    }

    private fun copyAssetToInternalStorageLocked(assetName: String): File {
        val output = File(appContext.filesDir, assetName)
        val expectedSize = runCatching { appContext.assets.openFd(assetName).declaredLength }
            .getOrNull()
        if (output.isFile && output.length() > 0L && (expectedSize == null || output.length() == expectedSize)) {
            return output
        }

        val temporary = File(appContext.filesDir, "$assetName.tmp")
        appContext.assets.open(assetName).use { input ->
            FileOutputStream(temporary, false).use { fileOutput ->
                input.copyTo(fileOutput, COPY_BUFFER_SIZE)
                fileOutput.fd.sync()
            }
        }
        check(temporary.length() > 0L) { "Copied model asset is empty: $assetName" }
        if (output.exists() && !output.delete()) {
            temporary.delete()
            error("Could not replace cached model: ${output.absolutePath}")
        }
        check(temporary.renameTo(output)) { "Could not finalize cached model: $assetName" }
        return output
    }

    companion object {
        const val YOLO_MODEL_ASSET = "RoEmotion_LED_Recognition_Model_YOLOv26n.tflite"
        const val EMOTION_MODEL_ASSET =
            "RoEmotion_Emotion_Detection_ResNet_50_LSTM_Attention.pt"

        private const val TAG = "InferenceModelLoader"
        private const val COPY_BUFFER_SIZE = 1024 * 1024
        private val ASSET_COPY_LOCK = Any()
    }
}

class YoloModelSession internal constructor(
    private val compiledModel: CompiledModel,
    val accelerator: String
) : Closeable {
    private val closed = AtomicBoolean(false)
    private val inputBuffers = compiledModel.createInputBuffers()
    private val outputBuffers = compiledModel.createOutputBuffers()
    val inputShape: IntArray = intArrayOf(1, YOLO_CHANNELS, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
    val outputShape: IntArray = intArrayOf(1, YOLO_MAX_DETECTIONS, YOLO_VALUES_PER_DETECTION)
    val inputWidth: Int = inputShape[3]
    val inputHeight: Int = inputShape[2]

    init {
        require(inputBuffers.size == 1) { "Expected one YOLO input tensor." }
        require(outputBuffers.size == 1) { "Expected one YOLO output tensor." }
    }

    @Synchronized
    fun run(input: FloatArray): FloatArray {
        check(!closed.get()) { "The YOLO model session is closed." }
        val expectedInputSize = inputShape.fold(1, Int::times)
        require(input.size == expectedInputSize) {
            "YOLO input has ${input.size} values; expected $expectedInputSize."
        }
        inputBuffers.single().writeFloat(input)
        compiledModel.run(inputBuffers, outputBuffers)
        return outputBuffers.single().readFloat().also { output ->
            val expectedOutputSize = outputShape.fold(1, Int::times)
            check(output.size == expectedOutputSize) {
                "YOLO output has ${output.size} values; expected $expectedOutputSize."
            }
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        try {
            inputBuffers.forEach { it.close() }
        } finally {
            try {
                outputBuffers.forEach { it.close() }
            } finally {
                compiledModel.close()
            }
        }
    }

    private companion object {
        const val YOLO_INPUT_SIZE = 960
        const val YOLO_CHANNELS = 3
        const val YOLO_MAX_DETECTIONS = 300
        const val YOLO_VALUES_PER_DETECTION = 6
    }
}

data class LoadedInferenceModels(
    val yoloModel: YoloModelSession,
    val emotionClassifier: PyTorchClassifier
) : Closeable {
    private val closed = AtomicBoolean(false)

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        try {
            yoloModel.close()
        } finally {
            emotionClassifier.close()
        }
    }
}
