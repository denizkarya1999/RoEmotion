package com.developer27.xemotion.inference

import android.app.AlertDialog
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.developer27.xemotion.R
import com.developer27.xemotion.inference.local.DatasetInput
import com.developer27.xemotion.inference.local.DatasetInputType
import com.developer27.xemotion.inference.local.DatasetKind
import com.developer27.xemotion.inference.local.LocalDatasetInputResolver
import com.developer27.xemotion.inference.local.LocalInferenceReportStore
import com.developer27.xemotion.inference.local.LocalInferenceViewModel
import com.developer27.xemotion.inference.local.PreparedDatasetRoot
import com.developer27.xemotion.inference.local.ResNetDatasetParser
import com.developer27.xemotion.inference.local.ResNetDatasetEvaluator
import com.developer27.xemotion.inference.local.YoloDatasetEvaluator
import com.developer27.xemotion.inference.local.YoloDatasetParser
import com.developer27.xemotion.ui.applySystemBarPadding
import com.developer27.xemotion.ui.enableRoEmotionEdgeToEdge
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

class LocalInferenceActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "LocalInferenceActivity"

        // Fixed YOLO input size used by the TFLite model
        private const val YOLO_INPUT_W = 640
        private const val YOLO_INPUT_H = 640

    }

    private lateinit var yoloDatasetPathEditText: EditText
    private lateinit var resnetDatasetPathEditText: EditText
    private lateinit var selectYoloDatasetButton: Button
    private lateinit var selectResnetDatasetButton: Button
    private lateinit var inferYoloButton: Button
    private lateinit var inferResnetButton: Button

    private lateinit var viewModel: LocalInferenceViewModel
    private val activityJob = SupervisorJob()
    private val activityScope = CoroutineScope(activityJob + Dispatchers.Main)
    private val modelLock = Any()
    private var yoloModel: YoloModelSession? = null
    private var pytorchClassifier: PyTorchClassifier? = null

    private val inputResolver by lazy { LocalDatasetInputResolver(this) }
    private val modelLoader by lazy { PyTorchModelLoader(applicationContext) }
    private val yoloDatasetParser = YoloDatasetParser()
    private val resNetDatasetParser = ResNetDatasetParser()
    private val yoloDatasetEvaluator = YoloDatasetEvaluator(yoloDatasetParser)
    private val resNetDatasetEvaluator = ResNetDatasetEvaluator(resNetDatasetParser)
    private val reportStore by lazy { LocalInferenceReportStore(this) }

    // Folder picker
    private val openDatasetTreeLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
            if (uri == null) return@registerForActivityResult

            try {
                // Persist read permission for future access
                contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (_: Exception) {
            }

            val kind = viewModel.pendingDatasetKind ?: return@registerForActivityResult
            val picked = DatasetInput(uri, DatasetInputType.FOLDER)
            viewModel.setInput(kind, picked)
            when (kind) {
                DatasetKind.YOLO -> {
                    yoloDatasetPathEditText.setText(uri.toString())
                }
                DatasetKind.RESNET -> {
                    resnetDatasetPathEditText.setText(uri.toString())
                }
            }
        }

    // ZIP file picker
    private val openZipFileLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri == null) return@registerForActivityResult

            try {
                // Persist read permission for future access
                contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (_: Exception) {
            }

            val kind = viewModel.pendingDatasetKind ?: return@registerForActivityResult
            val picked = DatasetInput(uri, DatasetInputType.ZIP)
            viewModel.setInput(kind, picked)
            when (kind) {
                DatasetKind.YOLO -> {
                    yoloDatasetPathEditText.setText(uri.toString())
                }
                DatasetKind.RESNET -> {
                    resnetDatasetPathEditText.setText(uri.toString())
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableRoEmotionEdgeToEdge()

        // Keep the device awake while processing a dataset.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.activity_local_inference)
        findViewById<View>(R.id.local_inference_root).applySystemBarPadding()

        viewModel = ViewModelProvider(this)[LocalInferenceViewModel::class.java]
        bindViews()
        setupClickListeners()
    }

    private fun bindViews() {
        yoloDatasetPathEditText = findViewById(R.id.yoloDatasetPathEditText)
        resnetDatasetPathEditText = findViewById(R.id.resnetDatasetPathEditText)
        selectYoloDatasetButton = findViewById(R.id.selectYoloDatasetButton)
        selectResnetDatasetButton = findViewById(R.id.selectResnetDatasetButton)
        inferYoloButton = findViewById(R.id.inferYoloButton)
        inferResnetButton = findViewById(R.id.inferResnetButton)
    }

    private fun setupClickListeners() {
        selectYoloDatasetButton.setOnClickListener {
            viewModel.pendingDatasetKind = DatasetKind.YOLO
            showInputPickerTypeDialog()
        }

        selectResnetDatasetButton.setOnClickListener {
            viewModel.pendingDatasetKind = DatasetKind.RESNET
            showInputPickerTypeDialog()
        }

        inferYoloButton.setOnClickListener {
            runSelectedDataset(DatasetKind.YOLO, inferYoloButton)
        }

        inferResnetButton.setOnClickListener {
            runSelectedDataset(DatasetKind.RESNET, inferResnetButton)
        }
    }

    private fun runSelectedDataset(kind: DatasetKind, button: Button) {
        val input = viewModel.inputFor(kind)
        if (input == null) {
            Toast.makeText(this, "Select a dataset first.", Toast.LENGTH_SHORT).show()
            return
        }

        button.isEnabled = false
        activityScope.launch {
            try {
                when (kind) {
                    DatasetKind.YOLO -> runYoloInferenceOverDataset(input)
                    DatasetKind.RESNET -> runResNetInferenceOverDataset(input)
                }
            } finally {
                button.isEnabled = true
            }
        }
    }

    // -------------------------------------------------------------------------
    // UI helpers
    // -------------------------------------------------------------------------

    private fun showInputPickerTypeDialog() {
        AlertDialog.Builder(this)
            .setTitle("Select Input Type")
            .setItems(arrayOf("Folder", "ZIP File")) { _, which ->
                when (which) {
                    0 -> {
                        openDatasetTreeLauncher.launch(null)
                    }

                    1 -> {
                        openZipFileLauncher.launch(arrayOf("application/zip", "application/octet-stream", "*/*"))
                    }
                }
            }
            .show()
    }

    private data class ProgressUi(
        val dialog: AlertDialog,
        val progressBar: ProgressBar,
        val statusTextView: TextView
    )

    private data class InferenceResult(
        val title: String,
        val fileName: String,
        val savedUri: Uri?,
        val reportText: String
    )

    private suspend fun createProgressDialog(title: String): ProgressUi = withContext(Dispatchers.Main) {
        val container = LinearLayout(this@LocalInferenceActivity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 40, 48, 16)
        }

        val titleView = TextView(this@LocalInferenceActivity).apply {
            text = title
            textSize = 18f
        }

        val progressBar = ProgressBar(
            this@LocalInferenceActivity,
            null,
            android.R.attr.progressBarStyleHorizontal
        ).apply {
            isIndeterminate = false
            max = 100
            progress = 0
        }

        val statusView = TextView(this@LocalInferenceActivity).apply {
            text = "Preparing..."
            setPadding(0, 24, 0, 0)
        }

        container.addView(titleView)
        container.addView(progressBar)
        container.addView(statusView)

        val dialog = AlertDialog.Builder(this@LocalInferenceActivity)
            .setView(container)
            .setCancelable(false)
            .create()

        dialog.show()
        ProgressUi(dialog, progressBar, statusView)
    }

    private suspend fun updateProgress(ui: ProgressUi, current: Int, total: Int, message: String) {
        withContext(Dispatchers.Main) {
            val progress = if (total <= 0) {
                0
            } else {
                ((current.toFloat() / total.toFloat()) * 100f).roundToInt()
            }

            ui.progressBar.progress = progress.coerceIn(0, 100)
            ui.statusTextView.text = "$message ($current / $total)"
        }
    }

    private suspend fun closeProgressDialog(ui: ProgressUi) {
        withContext(Dispatchers.Main) {
            if (ui.dialog.isShowing) ui.dialog.dismiss()
        }
    }

    private suspend fun showMessageBox(title: String, message: String) {
        withContext(Dispatchers.Main) {
            AlertDialog.Builder(this@LocalInferenceActivity)
                .setTitle(title)
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show()
        }
    }

    // -------------------------------------------------------------------------
    // Input preparation (Folder / ZIP -> local File root)
    // -------------------------------------------------------------------------

    private suspend fun prepareInputRoot(
        input: DatasetInput,
        kind: DatasetKind,
        progressUi: ProgressUi
    ): PreparedDatasetRoot = inputResolver.prepare(
        input = input,
        kind = kind,
        onProgress = { current, total, message ->
            updateProgress(progressUi, current, total, message)
        }
    )

    // -------------------------------------------------------------------------
    // YOLO dataset inference
    // -------------------------------------------------------------------------

    private suspend fun runYoloInferenceOverDataset(input: DatasetInput) {
        val progressUi = createProgressDialog("YOLO Dataset Inference")
        try {
            val result = withContext(Dispatchers.IO) {
                val prepared = prepareInputRoot(input, DatasetKind.YOLO, progressUi)
                try {
                    val model = getOrCreateYoloModel()
                    val report = yoloDatasetEvaluator.evaluate(
                        root = prepared.directory,
                        inputDescription = prepared.description,
                        predictor = { bitmap -> runSingleYoloInference(bitmap, model) },
                        onProgress = { current, total, message ->
                            updateProgress(progressUi, current, total, message)
                        }
                    )
                    val fileName = "yolo_inference_${reportStore.fileTimestamp()}.txt"
                    InferenceResult(
                        title = "YOLO Inference Finished",
                        fileName = fileName,
                        savedUri = reportStore.save(fileName, report),
                        reportText = report
                    )
                } finally {
                    prepared.cleanup()
                }
            }
            closeProgressDialog(progressUi)
            showMessageBox(
                result.title,
                reportStore.buildSummary(result.fileName, result.savedUri, result.reportText)
            )
        } catch (error: Exception) {
            Log.e(TAG, "YOLO inference failed", error)
            closeProgressDialog(progressUi)
            showMessageBox("YOLO Inference Error", error.message ?: "Unknown error")
        }
    }
    private fun runSingleYoloInference(bitmap: Bitmap, model: YoloModelSession): List<YoloDet> {
        // Resize/pad image to the model input size
        val meta = YoloLedDetection.createLetterboxedBitmap(
            srcBitmap = bitmap,
            targetWidth = YOLO_INPUT_W,
            targetHeight = YOLO_INPUT_H
        )

        return try {
            val shape = model.outputShape
            require(shape.size == 3) {
                "Unsupported YOLO output shape: ${shape.contentToString()}"
            }
            val input = YoloLedDetection.bitmapToNormalizedTensorNHWC(meta.inputBitmap)
            YoloLedDetection.parseTFLite(model.run(input), shape)
        } finally {
            if (!meta.inputBitmap.isRecycled) meta.inputBitmap.recycle()
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
    // -------------------------------------------------------------------------
    // ResNet dataset inference
    // -------------------------------------------------------------------------

    private suspend fun runResNetInferenceOverDataset(input: DatasetInput) {
        val progressUi = createProgressDialog("ResNet Dataset Inference")
        try {
            val result = withContext(Dispatchers.IO) {
                val prepared = prepareInputRoot(input, DatasetKind.RESNET, progressUi)
                try {
                    val classifier = getOrCreatePyTorchClassifier()
                    val report = resNetDatasetEvaluator.evaluate(
                        root = prepared.directory,
                        inputDescription = prepared.description,
                        predictor = classifier::classifySequence,
                        onProgress = { current, total, message ->
                            updateProgress(progressUi, current, total, message)
                        }
                    )
                    val fileName = "resnet_inference_${reportStore.fileTimestamp()}.txt"
                    InferenceResult(
                        title = "ResNet Inference Finished",
                        fileName = fileName,
                        savedUri = reportStore.save(fileName, report),
                        reportText = report
                    )
                } finally {
                    prepared.cleanup()
                }
            }
            closeProgressDialog(progressUi)
            showMessageBox(
                result.title,
                reportStore.buildSummary(result.fileName, result.savedUri, result.reportText)
            )
        } catch (error: Exception) {
            Log.e(TAG, "ResNet inference failed", error)
            closeProgressDialog(progressUi)
            showMessageBox("ResNet Inference Error", error.message ?: "Unknown error")
        }
    }
    private fun getOrCreatePyTorchClassifier(): PyTorchClassifier {
        synchronized(modelLock) { pytorchClassifier?.let { return it } }
        val loaded = modelLoader.loadEmotionClassifier()
        val retained = synchronized(modelLock) {
            when {
                !activityScope.isActive -> null
                pytorchClassifier != null -> pytorchClassifier
                else -> loaded.also { pytorchClassifier = it }
            }
        }
        if (retained !== loaded) loaded.close()
        return retained ?: throw CancellationException("Activity was destroyed")
    }

    override fun onDestroy() {
        activityJob.cancel()
        activityJob.invokeOnCompletion {
            synchronized(modelLock) {
                yoloModel?.close()
                yoloModel = null
                pytorchClassifier?.close()
                pytorchClassifier = null
            }
        }
        super.onDestroy()
    }

}
