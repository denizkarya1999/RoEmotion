package com.developer27.xemotion.videoprocessing

import android.content.SharedPreferences
import org.opencv.core.Scalar

/** Runtime configuration shared by camera, detection, drawing, and export pipelines. */
object Settings {
    data class ModeCapabilities(
        val traceCollectionSettingsEnabled: Boolean,
        val inferenceTraceSettingsEnabled: Boolean,
        val inferenceSettingsEnabled: Boolean,
        val inferenceActionsEnabled: Boolean
    )

    object OperatingMode {
        enum class Mode { DATA_COLLECTION, INFERENCE }

        @Volatile
        var current: Mode = Mode.INFERENCE
            private set

        internal fun update(mode: Mode) {
            current = mode
        }
    }

    object DetectionMode {
        enum class Mode { CONTOUR, YOLO }

        @Volatile
        var current: Mode = Mode.YOLO

        @Volatile
        var enableYOLOinference = true
    }

    object Inference {
        @Volatile
        var confidenceThreshold = 0.001f

        @Volatile
        var iouThreshold = 0.45f

        @Volatile
        var labelSize = DEFAULT_LABEL_SIZE
            private set

        val labelFontScale: Double
            get() = labelSize / 10.0

        fun updateLabelSize(value: Int) {
            labelSize = value.coerceIn(MIN_LABEL_SIZE, MAX_LABEL_SIZE)
        }

        const val MIN_LABEL_SIZE = 8
        const val MAX_LABEL_SIZE = 20
        const val DEFAULT_LABEL_SIZE = 16
    }

    object Trace {
        enum class Type(
            val displayName: String,
            val minimumPoints: Int
        ) {
            RAW("Raw", 2),
            RAW_CV("Raw + CV Processing", 2),
            SPLINE("Kalman-Filter", 3),
            SPLINE_CV("Kalman Filter + CV Processing", 3)
        }

        @Volatile
        var enableRAWtrace = false
            private set

        @Volatile
        var enableSPLINEtrace = true
            private set

        @Volatile
        var collectionTypes: Set<Type> = setOf(Type.SPLINE_CV)
            private set

        @Volatile
        var inferenceType: Type = Type.SPLINE_CV
            private set

        @Volatile
        var splineStep = 0.01

        var originalLineColor = Scalar(173.0, 216.0, 230.0)
        var splineLineColor = Scalar(255.0, 203.0, 5.0)

        @Volatile
        var boldness = DEFAULT_BOLDNESS
            private set

        val lineThickness: Int
            get() = boldness * OVERLAY_THICKNESS_MULTIPLIER

        val exportLineThickness: Int
            get() = boldness

        fun updateBoldness(value: Int) {
            boldness = value.coerceIn(MIN_BOLDNESS, MAX_BOLDNESS)
        }

        const val MIN_BOLDNESS = 1
        const val MAX_BOLDNESS = 20
        const val DEFAULT_BOLDNESS = 10
        private const val OVERLAY_THICKNESS_MULTIPLIER = 10

        fun updateCollectionTypes(types: Set<Type>) {
            require(types.isNotEmpty()) { "Select at least one trace type." }
            collectionTypes = types.toSet()
            enableRAWtrace = types.any { it == Type.RAW || it == Type.RAW_CV }
            enableSPLINEtrace = types.any { it == Type.SPLINE || it == Type.SPLINE_CV }
        }

        fun updateInferenceType(type: Type) {
            inferenceType = type
        }
    }

    object BoundingBox {
        enum class ColorMode { BY_USER, BY_EMOTION }

        @Volatile
        var enableBoundingBox = true

        var boxColor = Scalar(255.0, 255.0, 255.0)

        @Volatile
        var boxThickness = 10

        @Volatile
        var maxPerFrame = 3

        @Volatile
        var colorMode = ColorMode.BY_USER
    }

    object Brightness {
        @Volatile
        var factor = 2.0

        @Volatile
        var threshold = 150.0
    }

    object ExportData {
        @Volatile
        var frameIMG = false

        @Volatile
        var enablePredictionLogging = false
    }

    object RollingShutter {
        @Volatile
        var speedHz = 60f
    }

    /** Applies the fixed pipeline contract for the selected top-level mode. */
    fun applyOperatingMode(mode: OperatingMode.Mode) {
        OperatingMode.update(mode)
        when (mode) {
            OperatingMode.Mode.DATA_COLLECTION -> {
                DetectionMode.current = DetectionMode.Mode.CONTOUR
                DetectionMode.enableYOLOinference = false
                ExportData.frameIMG = true
            }

            OperatingMode.Mode.INFERENCE -> {
                DetectionMode.current = DetectionMode.Mode.YOLO
                DetectionMode.enableYOLOinference = true
                ExportData.frameIMG = false
            }
        }
    }

    fun capabilitiesFor(mode: OperatingMode.Mode): ModeCapabilities {
        val inferenceMode = mode == OperatingMode.Mode.INFERENCE
        return ModeCapabilities(
            traceCollectionSettingsEnabled = !inferenceMode,
            inferenceTraceSettingsEnabled = inferenceMode,
            inferenceSettingsEnabled = inferenceMode,
            inferenceActionsEnabled = inferenceMode
        )
    }

    /** Initializes runtime settings from the preferences persisted by SettingsActivity. */
    fun load(preferences: SharedPreferences) {
        val traceTypes = preferences.getStringSet(KEY_TRACE_TYPES, null)
            ?.mapNotNull { value -> runCatching { Trace.Type.valueOf(value) }.getOrNull() }
            ?.toSet()
            ?.takeIf { it.isNotEmpty() }
            ?: setOf(Trace.Type.SPLINE_CV)
        Trace.updateCollectionTypes(traceTypes)
        Trace.updateInferenceType(
            preferences.getString(KEY_INFERENCE_TRACE_TYPE, Trace.Type.SPLINE_CV.name)
                ?.let { value -> runCatching { Trace.Type.valueOf(value) }.getOrNull() }
                ?: Trace.Type.SPLINE_CV
        )
        Trace.updateBoldness(
            preferences.getInt(KEY_TRACE_BOLDNESS, Trace.DEFAULT_BOLDNESS)
        )
        Inference.updateLabelSize(
            preferences.getInt(KEY_INFERENCE_LABEL_SIZE, Inference.DEFAULT_LABEL_SIZE)
        )
        BoundingBox.enableBoundingBox = preferences.getBoolean(KEY_BOUNDING_BOX, true)
        BoundingBox.maxPerFrame = preferences.getString(KEY_MAX_BOXES, "3")
            ?.toIntOrNull()
            ?.coerceIn(1, 3)
            ?: 3
        ExportData.enablePredictionLogging = preferences.getBoolean(KEY_PREDICTION_LOGGING, false)
        RollingShutter.speedHz = preferences.getString(KEY_SHUTTER_SPEED, "60")
            ?.toFloatOrNull()
            ?: 60f

        val mode = runCatching {
            OperatingMode.Mode.valueOf(
                preferences.getString(KEY_OPERATING_MODE, OperatingMode.Mode.INFERENCE.name)
                    ?: OperatingMode.Mode.INFERENCE.name
            )
        }.getOrDefault(OperatingMode.Mode.INFERENCE)
        applyOperatingMode(mode)
    }

    const val KEY_OPERATING_MODE = "operating_mode"
    const val KEY_TRACE_TYPES = "trace_types"
    const val KEY_INFERENCE_TRACE_TYPE = "inference_trace_type"
    const val KEY_TRACE_BOLDNESS = "trace_boldness"
    const val KEY_INFERENCE_LABEL_SIZE = "inference_label_size"
    private const val KEY_BOUNDING_BOX = "enable_bounding_box"
    private const val KEY_MAX_BOXES = "max_boxes"
    private const val KEY_PREDICTION_LOGGING = "enable_prediction_logging"
    private const val KEY_SHUTTER_SPEED = "shutter_speed"
}
