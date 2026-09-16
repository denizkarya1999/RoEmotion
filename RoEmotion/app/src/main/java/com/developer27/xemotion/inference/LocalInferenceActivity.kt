package com.developer27.xemotion.inference

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.developer27.xemotion.R
import com.developer27.xemotion.inference.local.StandaloneInferenceViewModel
import com.developer27.xemotion.ui.applySystemBarPadding
import com.developer27.xemotion.ui.enableRoEmotionEdgeToEdge
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.util.Locale

/** Runs the bundled models over user-selected images without starting the camera. */
class LocalInferenceActivity : AppCompatActivity() {

    private lateinit var selectLedImagesButton: Button
    private lateinit var ledSelectionTextView: TextView
    private lateinit var ledGroundTruthSpinner: Spinner
    private lateinit var inferLedButton: Button
    private lateinit var ledResultsTextView: TextView
    private lateinit var selectEmotionImagesButton: Button
    private lateinit var emotionSelectionTextView: TextView
    private lateinit var emotionGroundTruthSpinner: Spinner
    private lateinit var inferEmotionButton: Button
    private lateinit var emotionResultsTextView: TextView

    private lateinit var viewModel: StandaloneInferenceViewModel
    private val activityJob = SupervisorJob()
    private val activityScope = CoroutineScope(activityJob + Dispatchers.Main)
    private val modelLock = Any()
    private var yoloModel: YoloModelSession? = null
    private var emotionClassifier: PyTorchClassifier? = null
    private lateinit var selectedEmotionModel: EmotionModelSpec

    private val modelLoader by lazy { PyTorchModelLoader(applicationContext) }
    private val openCvReady by lazy { OpenCVLoader.initLocal() }

    private val openLedImagesLauncher =
        registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isEmpty()) return@registerForActivityResult
            retainReadAccess(uris)
            viewModel.ledImageUris = uris
            viewModel.ledResults = ""
            renderState()
        }

    private val openEmotionImagesLauncher =
        registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isEmpty()) return@registerForActivityResult
            retainReadAccess(uris)
            viewModel.emotionImageUris = uris
            viewModel.emotionResults = ""
            renderState()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableRoEmotionEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_local_inference)
        findViewById<View>(R.id.local_inference_root).applySystemBarPadding()

        viewModel = ViewModelProvider(this)[StandaloneInferenceViewModel::class.java]
        bindViews()
        selectedEmotionModel = modelLoader.selectedModel()
        updateEmotionGroundTruthOptions()
        renderState()
        setupControls()
    }

    override fun onResume() {
        super.onResume()
        val selected = modelLoader.selectedModel()
        if (selected.id != selectedEmotionModel.id) {
            selectedEmotionModel = selected
            viewModel.emotionResults = ""
            updateEmotionGroundTruthOptions()
            renderState()
        }
    }

    private fun updateEmotionGroundTruthOptions() {
        if (viewModel.emotionGroundTruth !in selectedEmotionModel.labels) {
            viewModel.emotionGroundTruth = null
        }
        val prompt = resources.getStringArray(R.array.emotion_ground_truth_options).first()
        emotionGroundTruthSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            listOf(prompt) + selectedEmotionModel.labels
        ).apply { setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
    }

    private fun bindViews() {
        selectLedImagesButton = findViewById(R.id.selectLedImagesButton)
        ledSelectionTextView = findViewById(R.id.ledSelectionTextView)
        ledGroundTruthSpinner = findViewById(R.id.ledGroundTruthSpinner)
        inferLedButton = findViewById(R.id.inferLedButton)
        ledResultsTextView = findViewById(R.id.ledResultsTextView)
        selectEmotionImagesButton = findViewById(R.id.selectEmotionImagesButton)
        emotionSelectionTextView = findViewById(R.id.emotionSelectionTextView)
        emotionGroundTruthSpinner = findViewById(R.id.emotionGroundTruthSpinner)
        inferEmotionButton = findViewById(R.id.inferEmotionButton)
        emotionResultsTextView = findViewById(R.id.emotionResultsTextView)
    }

    private fun setupControls() {
        selectLedImagesButton.setOnClickListener {
            openLedImagesLauncher.launch(arrayOf("image/*"))
        }
        ledGroundTruthSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = (position - 1)
                    .takeIf { it in YoloLedDetection.userLabels.indices }
                if (selected != viewModel.ledGroundTruthClassId) {
                    viewModel.ledGroundTruthClassId = selected
                    viewModel.ledResults = ""
                    ledResultsTextView.text = getString(R.string.inference_results_placeholder)
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                viewModel.ledGroundTruthClassId = null
                viewModel.ledResults = ""
            }
        }
        inferLedButton.setOnClickListener { runLedInference() }

        selectEmotionImagesButton.setOnClickListener {
            openEmotionImagesLauncher.launch(arrayOf("image/*"))
        }
        emotionGroundTruthSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = selectedEmotionModel.labels.getOrNull(position - 1)
                if (selected != viewModel.emotionGroundTruth) {
                    viewModel.emotionGroundTruth = selected
                    viewModel.emotionResults = ""
                    emotionResultsTextView.text = getString(R.string.inference_results_placeholder)
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                viewModel.emotionGroundTruth = null
                viewModel.emotionResults = ""
            }
        }
        inferEmotionButton.setOnClickListener { runEmotionInference() }
    }

    private fun renderState() {
        ledSelectionTextView.text = selectionSummary(viewModel.ledImageUris)
        ledGroundTruthSpinner.setSelection(viewModel.ledGroundTruthClassId?.plus(1) ?: 0, false)
        ledResultsTextView.text = viewModel.ledResults.ifBlank {
            getString(R.string.inference_results_placeholder)
        }
        emotionSelectionTextView.text = selectionSummary(viewModel.emotionImageUris)
        emotionGroundTruthSpinner.setSelection(
            viewModel.emotionGroundTruth
                ?.let(selectedEmotionModel.labels::indexOf)
                ?.takeIf { it >= 0 }
                ?.plus(1)
                ?: 0,
            false
        )
        emotionResultsTextView.text = viewModel.emotionResults.ifBlank {
            getString(R.string.inference_results_placeholder)
        }
    }

    private fun runLedInference() {
        val groundTruth = viewModel.ledGroundTruthClassId
        if (viewModel.ledImageUris.isEmpty()) {
            toast(R.string.select_led_images_first)
            return
        }
        if (groundTruth == null) {
            toast(R.string.select_led_ground_truth_first)
            return
        }

        setControlsEnabled(false)
        ledResultsTextView.text = getString(R.string.loading_led_model)
        activityScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    require(openCvReady) { getString(R.string.opencv_initialization_failed) }
                    val model = getOrCreateYoloModel()
                    buildString {
                        appendLine("LED ID Detection Results")
                        appendLine("Ground truth for each image: ${YoloLedDetection.displayLabelForClass(groundTruth)}")
                        appendLine()
                        var exactMatches = 0
                        viewModel.ledImageUris.forEachIndexed { index, uri ->
                            withContext(Dispatchers.Main) {
                                ledResultsTextView.text = getString(
                                    R.string.processing_images_progress,
                                    index + 1,
                                    viewModel.ledImageUris.size
                                )
                            }
                            val bitmap = decodeBitmap(uri)
                                ?: throw IllegalArgumentException("Could not decode ${displayName(uri)}")
                            val detections = try {
                                YoloLedDetection.pickOnePerClass(runSingleYoloInference(bitmap, model))
                            } finally {
                                bitmap.recycle()
                            }
                            val predicted = detections.maxByOrNull(YoloDet::objConf)
                            val match = predicted?.classId == groundTruth
                            if (match) exactMatches++
                            appendLine(displayName(uri))
                            appendLine("  Ground truth: ${YoloLedDetection.displayLabelForClass(groundTruth)}")
                            appendLine("  Predicted: ${formatLedPrediction(predicted)}")
                            appendLine("  Match: ${if (match) "Yes" else "No"}")
                            appendLine()
                        }
                        appendLine("Matches: $exactMatches / ${viewModel.ledImageUris.size}")
                    }
                }
                viewModel.ledResults = result.trimEnd()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.e(TAG, "Standalone LED inference failed", error)
                viewModel.ledResults = getString(
                    R.string.inference_failed_with_reason,
                    error.message ?: getString(R.string.unknown_error)
                )
            } finally {
                if (activityScope.isActive) {
                    renderState()
                    setControlsEnabled(true)
                }
            }
        }
    }

    private fun runEmotionInference() {
        val requestedModel = selectedEmotionModel
        val groundTruth = viewModel.emotionGroundTruth
        if (viewModel.emotionImageUris.isEmpty()) {
            toast(R.string.select_emotion_images_first)
            return
        }
        if (groundTruth == null) {
            toast(R.string.select_emotion_ground_truth_first)
            return
        }

        setControlsEnabled(false)
        emotionResultsTextView.text = getString(R.string.loading_emotion_model)
        activityScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val bitmaps = mutableListOf<Bitmap>()
                    try {
                        val classifier = getOrCreateEmotionClassifier(requestedModel)
                        val sequenceUris = sampleEmotionSequence(
                            viewModel.emotionImageUris,
                            classifier.modelSpec.sequenceLength,
                            padShortSequence = classifier.modelSpec.isLegacy
                        )
                        sequenceUris.forEach { uri ->
                            bitmaps += decodeBitmap(uri)
                                ?: throw IllegalArgumentException("Could not decode ${displayName(uri)}")
                        }
                        val (prediction, probabilities) = classifier.classifySequence(bitmaps)
                        val bestIndex = probabilities.indices.maxByOrNull(probabilities::get) ?: 0
                        val confidence = probabilities.getOrElse(bestIndex) { 0f }
                        buildString {
                            appendLine("Emotion Recognition Results")
                            appendLine("Model: ${classifier.modelSpec.selectionSummary.replace("\n", " / ")}")
                            appendLine("Images selected: ${viewModel.emotionImageUris.size}")
                            appendLine("Frames used by model: ${bitmaps.size}")
                            appendLine("Ground truth: $groundTruth")
                            appendLine("Predicted: $prediction (${formatPercent(confidence)})")
                            appendLine("Match: ${if (groundTruth.equals(prediction, true)) "Yes" else "No"}")
                            appendLine()
                            appendLine("Probabilities")
                            classifier.labels.forEachIndexed { index, label ->
                                appendLine("  $label: ${formatPercent(probabilities.getOrElse(index) { 0f })}")
                            }
                        }
                    } finally {
                        bitmaps.forEach(Bitmap::recycle)
                    }
                }
                if (requestedModel.id == selectedEmotionModel.id) {
                    viewModel.emotionResults = result.trimEnd()
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.e(TAG, "Standalone emotion inference failed", error)
                if (requestedModel.id == selectedEmotionModel.id) {
                    viewModel.emotionResults = getString(
                        R.string.inference_failed_with_reason,
                        error.message ?: getString(R.string.unknown_error)
                    )
                }
            } finally {
                if (activityScope.isActive) {
                    renderState()
                    setControlsEnabled(true)
                }
            }
        }
    }

    private fun runSingleYoloInference(bitmap: Bitmap, model: YoloModelSession): List<YoloDet> {
        val meta = YoloLedDetection.createLetterboxedBitmap(
            srcBitmap = bitmap,
            targetWidth = model.inputWidth,
            targetHeight = model.inputHeight
        )
        return try {
            val input = YoloLedDetection.bitmapToNormalizedTensorNCHW(meta.inputBitmap)
            YoloLedDetection.parseTFLite(model.run(input), model.outputShape)
        } finally {
            meta.inputBitmap.recycle()
        }
    }

    private fun getOrCreateYoloModel(): YoloModelSession {
        synchronized(modelLock) { yoloModel?.let { return it } }
        val loaded = modelLoader.loadYoloModel()
        val retained = synchronized(modelLock) {
            when {
                !activityScope.isActive -> null
                yoloModel != null -> yoloModel
                else -> loaded.also { yoloModel = it }
            }
        }
        if (retained !== loaded) loaded.close()
        return retained ?: throw CancellationException("Activity was destroyed")
    }

    private fun getOrCreateEmotionClassifier(spec: EmotionModelSpec): PyTorchClassifier {
        synchronized(modelLock) {
            emotionClassifier?.takeIf { it.modelSpec.id == spec.id }?.let { return it }
            emotionClassifier?.close()
            emotionClassifier = null
        }
        val loaded = modelLoader.loadEmotionClassifier(spec)
        val retained = synchronized(modelLock) {
            when {
                !activityScope.isActive -> null
                emotionClassifier != null -> emotionClassifier
                else -> loaded.also { emotionClassifier = it }
            }
        }
        if (retained !== loaded) loaded.close()
        return retained ?: throw CancellationException("Activity was destroyed")
    }

    private fun decodeBitmap(uri: Uri): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var sampleSize = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sampleSize > MAX_DECODED_IMAGE_EDGE) {
            sampleSize *= 2
        }
        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        }
    }

    private fun selectionSummary(uris: List<Uri>): String = when (uris.size) {
        0 -> getString(R.string.no_images_selected)
        1 -> getString(R.string.one_image_selected, displayName(uris.single()))
        else -> resources.getQuantityString(R.plurals.images_selected, uris.size, uris.size)
    }

    private fun displayName(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) {
                    cursor.getString(index)?.takeIf(String::isNotBlank)?.let { return it }
                }
            }
        }
        return uri.lastPathSegment ?: uri.toString()
    }

    private fun formatLedPrediction(detection: YoloDet?): String = detection?.let {
        "${YoloLedDetection.displayLabelForClass(it.classId)} ${formatPercent(it.objConf)}"
    } ?: getString(R.string.no_led_detected)

    private fun formatPercent(value: Float): String =
        String.format(Locale.US, "%.2f%%", value.coerceIn(0f, 1f) * 100f)

    private fun retainReadAccess(uris: List<Uri>) {
        uris.forEach { uri ->
            runCatching {
                contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
        }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        selectLedImagesButton.isEnabled = enabled
        ledGroundTruthSpinner.isEnabled = enabled
        inferLedButton.isEnabled = enabled
        selectEmotionImagesButton.isEnabled = enabled
        emotionGroundTruthSpinner.isEnabled = enabled
        inferEmotionButton.isEnabled = enabled
    }

    private fun toast(message: Int) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        activityJob.cancel()
        activityJob.invokeOnCompletion {
            synchronized(modelLock) {
                yoloModel?.close()
                yoloModel = null
                emotionClassifier?.close()
                emotionClassifier = null
            }
        }
        super.onDestroy()
    }

    private companion object {
        const val TAG = "StandaloneInference"
        const val MAX_DECODED_IMAGE_EDGE = 2_048
    }
}
