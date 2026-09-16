package com.developer27.xemotion.inference

import android.content.SharedPreferences
import com.developer27.xemotion.videoprocessing.Settings

enum class EmotionTrainingDistribution(val displayName: String, val assetKey: String) {
    RANDOM("Trained with Random Distribution", "random"),
    DIFFERENT_PERSON("Trained with Different Person Validation", "person")
}

enum class EmotionModelArchitecture(val displayName: String, val assetKey: String) {
    STANDALONE("Standalone ResNet-50", "standalone"),
    LSTM("ResNet-50 + LSTM", "lstm"),
    LSTM_ATTENTION("ResNet-50 + LSTM + Attention", "lstm_attention")
}

enum class EmotionTraceVariant(
    val displayName: String,
    val assetKey: String,
    val traceType: Settings.Trace.Type
) {
    RAW("Raw", "raw", Settings.Trace.Type.RAW),
    KALMAN_CV("Kalman Filter + CV", "kalman_cv", Settings.Trace.Type.SPLINE_CV)
}

/** The complete input/output contract of a bundled emotion model. */
data class EmotionModelSpec(
    val id: String,
    val displayName: String,
    val assetName: String,
    val labels: List<String>,
    val sequenceLength: Int,
    val isLegacy: Boolean,
    val trainingDistribution: EmotionTrainingDistribution?,
    val architecture: EmotionModelArchitecture,
    val traceVariant: EmotionTraceVariant?
) {
    val requiredTraceType: Settings.Trace.Type?
        get() = traceVariant?.traceType

    val selectionSummary: String
        get() = if (isLegacy) {
            "Previous Models\n$displayName"
        } else {
            "Current Models\n${trainingDistribution!!.displayName}\n$displayName\n${traceVariant!!.displayName}"
        }
}

/** Stable persisted IDs retain the original model for existing installations. */
object EmotionModelCatalog {
    const val KEY_SELECTION = "emotion_detection_model"

    val legacy = EmotionModelSpec(
        id = "previous_resnet50_lstm_attention",
        displayName = "ResNet-50 + LSTM + Attention Model",
        assetName = "RoEmotion_Emotion_Detection_ResNet_50_LSTM_Attention.pt",
        labels = listOf("Angry", "Anxiety", "Disgust", "Excitement", "Sadness"),
        sequenceLength = 5,
        isLegacy = true,
        trainingDistribution = null,
        architecture = EmotionModelArchitecture.LSTM_ATTENTION,
        traceVariant = null
    )

    private val currentLabels = listOf(
        "Amusement", "Anger", "Anxiety", "Disgust", "Fear",
        "Joy", "Neutral", "Sadness", "Surprise"
    )

    val models: List<EmotionModelSpec> = listOf(legacy) +
        EmotionTrainingDistribution.entries.flatMap { distribution ->
            EmotionModelArchitecture.entries.flatMap { architecture ->
                EmotionTraceVariant.entries.map { trace ->
                    val id = "current_${distribution.assetKey}_${architecture.assetKey}_${trace.assetKey}"
                    EmotionModelSpec(
                        id = id,
                        displayName = architecture.displayName,
                        assetName = "$id.pt",
                        labels = currentLabels,
                        sequenceLength = if (architecture == EmotionModelArchitecture.STANDALONE) 1 else 5,
                        isLegacy = false,
                        trainingDistribution = distribution,
                        architecture = architecture,
                        traceVariant = trace
                    )
                }
            }
        }

    fun findById(id: String?): EmotionModelSpec? = models.firstOrNull { it.id == id }

    fun selected(preferences: SharedPreferences): EmotionModelSpec =
        findById(preferences.getString(KEY_SELECTION, legacy.id)) ?: legacy

    /** Called only after the user chooses a leaf in the model picker. */
    fun select(preferences: SharedPreferences, model: EmotionModelSpec) {
        require(findById(model.id) == model) { "Unknown emotion model: ${model.id}" }
        val editor = preferences.edit().putString(KEY_SELECTION, model.id)
        model.requiredTraceType?.let { traceType ->
            editor.putString(Settings.KEY_INFERENCE_TRACE_TYPE, traceType.name)
        }
        editor.apply()
        model.requiredTraceType?.let(Settings.Trace::updateInferenceType)
    }
}
