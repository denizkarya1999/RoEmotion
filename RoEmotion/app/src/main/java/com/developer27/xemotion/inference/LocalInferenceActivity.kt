package com.developer27.xemotion.inference

import android.app.AlertDialog
import android.content.ContentValues
import android.content.Context
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.lifecycleScope
import com.developer27.xemotion.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import kotlin.math.roundToInt

class LocalInferenceActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "LocalInferenceActivity"

        // Picker mode for deciding whether current selection is YOLO or ResNet
        private const val PICK_MODE_YOLO = 1
        private const val PICK_MODE_RESNET = 2

        // Input can be either a folder tree or a zip file
        private const val INPUT_TYPE_TREE = 1
        private const val INPUT_TYPE_ZIP = 2

        // Fixed YOLO input size used by the TFLite model
        private const val YOLO_INPUT_W = 640
        private const val YOLO_INPUT_H = 640

        // ResNet sequence model uses a fixed number of frames
        private const val RESNET_SEQUENCE_LENGTH = 5
    }

    private lateinit var yoloDatasetPathEditText: EditText
    private lateinit var resnetDatasetPathEditText: EditText
    private lateinit var selectYoloDatasetButton: Button
    private lateinit var selectResnetDatasetButton: Button
    private lateinit var inferYoloButton: Button
    private lateinit var inferResnetButton: Button

    private var currentPickMode: Int = 0
    private var currentInputType: Int = 0

    private data class PickedInput(
        val uri: Uri,
        val type: Int
    )

    private var yoloInput: PickedInput? = null
    private var resnetInput: PickedInput? = null

    // Cached model instances
    private var yoloInterpreter: Interpreter? = null
    private var pytorchClassifier: PyTorchClassifier? = null

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

            val picked = PickedInput(uri = uri, type = INPUT_TYPE_TREE)
            when (currentPickMode) {
                PICK_MODE_YOLO -> {
                    yoloInput = picked
                    yoloDatasetPathEditText.setText(uri.toString())
                }

                PICK_MODE_RESNET -> {
                    resnetInput = picked
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

            val picked = PickedInput(uri = uri, type = INPUT_TYPE_ZIP)
            when (currentPickMode) {
                PICK_MODE_YOLO -> {
                    yoloInput = picked
                    yoloDatasetPathEditText.setText(uri.toString())
                }

                PICK_MODE_RESNET -> {
                    resnetInput = picked
                    resnetDatasetPathEditText.setText(uri.toString())
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Keep device awake and lock activity to portrait
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        setContentView(R.layout.activity_local_inference)

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
            currentPickMode = PICK_MODE_YOLO
            showInputPickerTypeDialog()
        }

        selectResnetDatasetButton.setOnClickListener {
            currentPickMode = PICK_MODE_RESNET
            showInputPickerTypeDialog()
        }

        inferYoloButton.setOnClickListener {
            val input = yoloInput
            if (input == null) {
                Toast.makeText(this, "Please select the YOLO dataset ZIP or folder first.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            lifecycleScope.launch(Dispatchers.Main) {
                runYoloInferenceOverDataset(input)
            }
        }

        inferResnetButton.setOnClickListener {
            val input = resnetInput
            if (input == null) {
                Toast.makeText(this, "Please select the ResNet dataset ZIP or folder first.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            lifecycleScope.launch(Dispatchers.Main) {
                runResNetInferenceOverDataset(input)
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
                        currentInputType = INPUT_TYPE_TREE
                        openDatasetTreeLauncher.launch(null)
                    }

                    1 -> {
                        currentInputType = INPUT_TYPE_ZIP
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

    private data class PreparedRoot(
        val rootDir: File,
        val description: String
    )

    private suspend fun prepareInputRoot(
        pickedInput: PickedInput,
        forMode: Int,
        progressUi: ProgressUi
    ): PreparedRoot = withContext(Dispatchers.IO) {
        // Temporary working folder used for copied or extracted data
        val tempRoot = File(cacheDir, "local_inference_inputs").apply { mkdirs() }
        val workDir = File(tempRoot, "${System.currentTimeMillis()}_${UUID.randomUUID()}").apply { mkdirs() }

        when (pickedInput.type) {
            INPUT_TYPE_ZIP -> {
                updateProgress(progressUi, 1, 4, "Extracting ZIP")
                unzipUriToDirectory(pickedInput.uri, workDir)
            }

            INPUT_TYPE_TREE -> {
                updateProgress(progressUi, 1, 4, "Copying folder")
                copyDocumentTreeToDirectory(pickedInput.uri, workDir)
            }

            else -> throw IllegalStateException("Unsupported input type.")
        }

        updateProgress(progressUi, 2, 4, "Locating dataset root")

        // Detect the actual dataset root after extraction/copy
        val detectedRoot = when (forMode) {
            PICK_MODE_YOLO -> findLikelyYoloDatasetRoot(workDir)
            PICK_MODE_RESNET -> findLikelyResNetDatasetRoot(workDir)
            else -> workDir
        } ?: throw IllegalStateException("Could not detect a valid dataset root inside the selected input.")

        PreparedRoot(
            rootDir = detectedRoot,
            description = pickedInput.uri.toString()
        )
    }

    private fun unzipUriToDirectory(zipUri: Uri, outDir: File) {
        contentResolver.openInputStream(zipUri)?.use { input ->
            ZipInputStream(BufferedInputStream(input)).use { zis ->
                var entry: ZipEntry? = zis.nextEntry
                while (entry != null) {
                    val safeFile = resolveZipEntrySafely(outDir, entry.name)

                    if (entry.isDirectory) {
                        safeFile.mkdirs()
                    } else {
                        safeFile.parentFile?.mkdirs()
                        FileOutputStream(safeFile).use { fos ->
                            zis.copyTo(fos)
                        }
                    }

                    zis.closeEntry()
                    entry = zis.nextEntry
                }
            }
        } ?: throw IllegalStateException("Failed to open ZIP file.")
    }

    private fun resolveZipEntrySafely(destinationDir: File, entryName: String): File {
        val outFile = File(destinationDir, entryName)
        val destPath = destinationDir.canonicalPath
        val outPath = outFile.canonicalPath

        // Prevent zip-slip path traversal
        if (!outPath.startsWith(destPath + File.separator) && outPath != destPath) {
            throw SecurityException("Blocked unsafe ZIP entry: $entryName")
        }
        return outFile
    }

    private fun copyDocumentTreeToDirectory(treeUri: Uri, outDir: File) {
        val rootDoc = DocumentFile.fromTreeUri(this, treeUri)
            ?: throw IllegalStateException("Could not open selected folder.")

        copyDocumentFileRecursive(rootDoc, outDir)
    }

    private fun copyDocumentFileRecursive(doc: DocumentFile, destination: File) {
        if (doc.isDirectory) {
            val dir = if (doc.name.isNullOrBlank()) destination else File(destination, doc.name!!)
            dir.mkdirs()
            doc.listFiles().forEach { child ->
                copyDocumentFileRecursive(child, dir)
            }
        } else if (doc.isFile) {
            val outFile = File(destination, doc.name ?: "unknown_file")
            doc.uri.let { uri ->
                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(outFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
        }
    }

    private fun findLikelyYoloDatasetRoot(startDir: File): File? {
        if (isYoloDatasetRoot(startDir)) return startDir

        val allDirs = startDir.walkTopDown()
            .filter { it.isDirectory }
            .toList()

        return allDirs.firstOrNull { isYoloDatasetRoot(it) }
    }

    private fun isYoloDatasetRoot(dir: File): Boolean {
        // YOLO root must have a yaml file with class/split information
        val yamlExists = File(dir, "data.yaml").exists() ||
                File(dir, "dataset.yaml").exists() ||
                dir.listFiles()?.any {
                    it.isFile && (it.name.endsWith(".yaml", true) || it.name.endsWith(".yml", true))
                } == true

        if (!yamlExists) return false

        // Roboflow style: train/images, valid/images, test/images
        val hasRoboflowStyleSplits = listOf("train", "valid", "val", "test").any { split ->
            File(dir, "$split/images").isDirectory
        }

        // Classic style: images/train, images/val, images/test
        val hasClassicYoloStyleSplits =
            File(dir, "images").isDirectory &&
                    listOf("train", "valid", "val", "test").any { split ->
                        File(dir, "images/$split").isDirectory
                    }

        return hasRoboflowStyleSplits || hasClassicYoloStyleSplits
    }

    private fun findLikelyResNetDatasetRoot(startDir: File): File? {
        if (containsImageFilesRecursively(startDir)) return startDir

        val allDirs = startDir.walkTopDown()
            .filter { it.isDirectory }
            .toList()

        return allDirs.firstOrNull { containsImageFilesRecursively(it) }
    }

    private fun containsImageFilesRecursively(dir: File): Boolean {
        return dir.walkTopDown().any { it.isFile && isImageFileName(it.name) }
    }

    // -------------------------------------------------------------------------
    // YOLO dataset inference
    // -------------------------------------------------------------------------

    private suspend fun runYoloInferenceOverDataset(input: PickedInput) {
        val progressUi = createProgressDialog("YOLO v26 Inference")

        try {
            val result = withContext(Dispatchers.IO) {
                val prepared = prepareInputRoot(input, PICK_MODE_YOLO, progressUi)
                val datasetRoot = prepared.rootDir

                updateProgress(progressUi, 3, 4, "Reading dataset")

                // Read YAML once, then use it for class names and split paths
                val yamlText = readBestYoloYamlText(datasetRoot)
                    ?: throw IllegalStateException("Could not read data.yaml / dataset.yaml.")

                val classNames = parseYoloClassNamesFromYamlText(yamlText)
                if (classNames.isEmpty()) {
                    throw IllegalStateException("Could not find class names from data.yaml / dataset.yaml.")
                }

                val splitInfos = collectYoloSplitFiles(datasetRoot, yamlText)
                val totalImages = splitInfos.sumOf { it.imageFiles.size }
                if (totalImages == 0) {
                    throw IllegalStateException("No YOLO images were found for train/val/test.")
                }

                val interpreter = getOrCreateYoloInterpreter()

                // Per-class statistics
                val perClassGt = IntArray(classNames.size)
                val perClassCorrect = IntArray(classNames.size)
                val imageLevelCorrect = AtomicInteger(0)

                // Per-split statistics
                val splitGtCounts = linkedMapOf<String, Int>()
                val splitCorrectCounts = linkedMapOf<String, Int>()
                val splitImageFullMatchCounts = linkedMapOf<String, Int>()

                val sb = StringBuilder()
                sb.appendLine("RoEmotion - YOLO v26 Dataset Inference Report")
                sb.appendLine("Generated: ${timestampNow()}")
                sb.appendLine("Selected Input: ${prepared.description}")
                sb.appendLine("Detected Dataset Root: ${datasetRoot.absolutePath}")
                sb.appendLine("Class Names: ${classNames.joinToString(", ")}")
                sb.appendLine()

                var processed = 0

                for (split in splitInfos) {
                    sb.appendLine("========== Split: ${split.splitName} ==========")

                    var splitGt = 0
                    var splitCorrect = 0
                    var splitImageFull = 0

                    for (imageFile in split.imageFiles) {
                        processed++
                        updateProgress(progressUi, processed, totalImages, "YOLO inferring ${split.splitName}")

                        // Ground-truth class ids from YOLO label txt
                        val gtClassIds = readYoloGroundTruthClassIds(split.labelsFolder, imageFile)
                        gtClassIds.forEach { classId ->
                            if (classId in perClassGt.indices) {
                                perClassGt[classId]++
                                splitGt++
                            }
                        }

                        val bitmap = loadBitmapFromFile(imageFile) ?: run {
                            sb.appendLine(imageFile.name)
                            sb.appendLine("  GT: ${gtClassIds.joinToString { classNames.getOrElse(it) { "Class_$it" } }}")
                            sb.appendLine("  Pred: [bitmap load failed]")
                            sb.appendLine()
                            continue
                        }

                        // Prediction returns multiple detections, so keep only the class
                        // with the highest confidence score for this frame
                        val detections = runSingleYoloInference(bitmap, interpreter)

                        val bestDetection = detections.maxByOrNull { it.objConf }

                        val predictedClassIds = bestDetection
                            ?.let { listOf(it.classId) }
                            ?: emptyList()

                        val imageGtSet = gtClassIds.toSet()
                        val imagePredSet = predictedClassIds.toSet()

                        // Image is counted as full match only if all GT classes are predicted
                        var imageMatchedAll = imageGtSet.isNotEmpty()

                        for (gtClass in imageGtSet) {
                            if (gtClass in imagePredSet) {
                                if (gtClass in perClassCorrect.indices) {
                                    perClassCorrect[gtClass]++
                                    splitCorrect++
                                }
                            } else {
                                imageMatchedAll = false
                            }
                        }

                        if (imageMatchedAll) {
                            imageLevelCorrect.incrementAndGet()
                            splitImageFull++
                        }

                        sb.appendLine(imageFile.name)
                        sb.appendLine("  GT: ${if (imageGtSet.isEmpty()) "[none]" else imageGtSet.joinToString { classNames.getOrElse(it) { "Class_$it" } }}")
                        sb.appendLine("  Pred: ${if (imagePredSet.isEmpty()) "[none]" else imagePredSet.joinToString { classNames.getOrElse(it) { "Class_$it" } }}")
                        sb.appendLine()
                    }

                    splitGtCounts[split.splitName] = splitGt
                    splitCorrectCounts[split.splitName] = splitCorrect
                    splitImageFullMatchCounts[split.splitName] = splitImageFull

                    sb.appendLine()
                }

                sb.appendLine("========== Per-Class Summary ==========")
                val totalGt = perClassGt.sum()
                val totalCorrect = perClassCorrect.sum()

                for (i in classNames.indices) {
                    val gt = perClassGt[i]
                    val correct = perClassCorrect[i]
                    val pct = if (gt == 0) 0.0 else (correct * 100.0 / gt.toDouble())
                    sb.appendLine(
                        "${classNames[i]} -> Ground Truth: $gt, Successful: $correct, Percentage: ${
                            String.format(Locale.US, "%.2f", pct)
                        }%"
                    )
                }

                sb.appendLine()
                sb.appendLine("========== Split Summary ==========")
                for (split in splitInfos) {
                    val splitName = split.splitName
                    val gt = splitGtCounts[splitName] ?: 0
                    val correct = splitCorrectCounts[splitName] ?: 0
                    val imageCount = split.imageFiles.size
                    val imageFull = splitImageFullMatchCounts[splitName] ?: 0

                    val gtPct = if (gt == 0) 0.0 else (correct * 100.0 / gt.toDouble())
                    val imagePct = if (imageCount == 0) 0.0 else (imageFull * 100.0 / imageCount.toDouble())

                    sb.appendLine("$splitName ->")
                    sb.appendLine("  Ground Truth Count: $gt")
                    sb.appendLine("  Successful Count: $correct")
                    sb.appendLine("  Success Percentage: ${String.format(Locale.US, "%.2f", gtPct)}%")
                    sb.appendLine("  Image-Level Full Match Count: $imageFull / $imageCount")
                    sb.appendLine("  Image-Level Full Match Percentage: ${String.format(Locale.US, "%.2f", imagePct)}%")
                }

                val overallPct = if (totalGt == 0) 0.0 else (totalCorrect * 100.0 / totalGt.toDouble())
                val imagePct = if (totalImages == 0) 0.0 else (imageLevelCorrect.get() * 100.0 / totalImages.toDouble())

                sb.appendLine()
                sb.appendLine("========== Overall Summary ==========")
                sb.appendLine("Overall Ground Truth Count: $totalGt")
                sb.appendLine("Overall Successful Count: $totalCorrect")
                sb.appendLine("Overall Success Percentage: ${String.format(Locale.US, "%.2f", overallPct)}%")
                sb.appendLine("Image-Level Full Match Count: ${imageLevelCorrect.get()} / $totalImages")
                sb.appendLine("Image-Level Full Match Percentage: ${String.format(Locale.US, "%.2f", imagePct)}%")

                val fileName = "yolo_v26_inference_${fileSafeTimestamp()}.txt"
                val savedUri = saveTextLogToDocuments(fileName, sb.toString())

                InferenceResult(
                    title = "YOLO Inference Finished",
                    fileName = fileName,
                    savedUri = savedUri,
                    reportText = sb.toString()
                )
            }

            closeProgressDialog(progressUi)

            showMessageBox(
                title = result.title,
                message = buildSavedLogSummary(
                    fileName = result.fileName,
                    savedUri = result.savedUri,
                    reportText = result.reportText
                )
            )
        } catch (e: Exception) {
            Log.e(TAG, "YOLO inference failed", e)
            closeProgressDialog(progressUi)
            showMessageBox("YOLO Inference Error", e.message ?: "Unknown error")
        }
    }

    private data class YoloSplitInfo(
        val splitName: String,
        val imagesFolder: File,
        val labelsFolder: File?,
        val imageFiles: List<File>
    )

    private fun readBestYoloYamlText(root: File): String? {
        val yaml = listOf(
            File(root, "data.yaml"),
            File(root, "dataset.yaml")
        ).firstOrNull { it.exists() }
            ?: root.listFiles()?.firstOrNull {
                it.isFile && (it.name.endsWith(".yaml", true) || it.name.endsWith(".yml", true))
            }
            ?: return null

        return runCatching { yaml.readText() }.getOrNull()
    }

    private fun parseYoloClassNamesFromYamlText(yamlText: String): List<String> {
        return parseYamlNames(yamlText)
    }

    private fun collectYoloSplitFiles(root: File, yamlText: String): List<YoloSplitInfo> {
        // Split paths can come from YAML or from common folder patterns
        val yamlSplitPaths = parseYoloSplitPathsFromYaml(yamlText)

        val orderedSplits = listOf("train", "val", "valid", "validation", "test", "testing")
        val results = mutableListOf<YoloSplitInfo>()
        val addedCanonicalPaths = mutableSetOf<String>()

        for (splitName in orderedSplits) {
            val candidateImageFolders = mutableListOf<File>()

            yamlSplitPaths[splitName]?.let { yamlPath ->
                resolveYoloSplitPath(root, yamlPath)?.let { candidateImageFolders += it }
            }

            when (splitName) {
                "train" -> {
                    candidateImageFolders += File(root, "train/images")
                    candidateImageFolders += File(root, "images/train")
                }

                "val", "valid", "validation" -> {
                    candidateImageFolders += File(root, "valid/images")
                    candidateImageFolders += File(root, "val/images")
                    candidateImageFolders += File(root, "validation/images")
                    candidateImageFolders += File(root, "images/valid")
                    candidateImageFolders += File(root, "images/val")
                    candidateImageFolders += File(root, "images/validation")
                }

                "test", "testing" -> {
                    candidateImageFolders += File(root, "test/images")
                    candidateImageFolders += File(root, "testing/images")
                    candidateImageFolders += File(root, "images/test")
                    candidateImageFolders += File(root, "images/testing")
                }
            }

            val imageFolder = candidateImageFolders
                .mapNotNull { candidate ->
                    runCatching { candidate.canonicalFile }.getOrNull()
                }
                .firstOrNull { it.exists() && it.isDirectory }

            if (imageFolder != null) {
                val canonical = imageFolder.canonicalPath
                if (canonical in addedCanonicalPaths) continue
                addedCanonicalPaths += canonical

                val labelsFolder = findLabelsFolderForImagesFolder(imageFolder)
                val imageFiles = imageFolder.listFiles()
                    ?.filter { it.isFile && isImageFileName(it.name) }
                    ?.sortedBy { it.name.lowercase(Locale.US) }
                    .orEmpty()

                if (imageFiles.isNotEmpty()) {
                    results += YoloSplitInfo(
                        splitName = normalizeSplitDisplayName(splitName, imageFolder),
                        imagesFolder = imageFolder,
                        labelsFolder = labelsFolder,
                        imageFiles = imageFiles
                    )
                }
            }
        }

        return results
    }

    private fun parseYoloSplitPathsFromYaml(yamlText: String): Map<String, String> {
        val result = mutableMapOf<String, String>()
        val regex = Regex("""^\s*(train|val|test)\s*:\s*(.+?)\s*$""", RegexOption.IGNORE_CASE)

        yamlText.lines().forEach { rawLine ->
            // Ignore YAML comments when parsing split paths
            val line = rawLine.substringBefore("#").trim()
            val match = regex.find(line) ?: return@forEach
            val key = match.groupValues[1].lowercase(Locale.US)
            val value = match.groupValues[2].trim().trim('"', '\'')
            if (value.isNotBlank()) {
                result[key] = value
            }
        }

        return result
    }

    private fun resolveYoloSplitPath(root: File, rawPath: String): File? {
        val cleaned = rawPath
            .replace("\\", "/")
            .trim()
            .trim('"', '\'')

        if (cleaned.isBlank()) return null

        // Normalize simple relative prefixes
        val normalizedRelative = cleaned
            .removePrefix("./")
            .removePrefix("../")
            .removePrefix("./")
            .removePrefix("../")

        val candidates = listOf(
            File(root, normalizedRelative),
            File(root, cleaned),
            File(root.parentFile, normalizedRelative)
        )

        return candidates.firstOrNull { it.exists() && it.isDirectory }
    }

    private fun findLabelsFolderForImagesFolder(imagesFolder: File): File? {
        val name = imagesFolder.name.lowercase(Locale.US)
        val parent = imagesFolder.parentFile ?: return null

        // For split/images -> split/labels
        if (name == "images") {
            val sibling = File(parent, "labels")
            if (sibling.exists() && sibling.isDirectory) return sibling
        }

        val directSibling = File(imagesFolder.parentFile ?: return null, "labels")
        if (directSibling.exists() && directSibling.isDirectory) return directSibling

        val guessed = File(parent, "labels")
        return guessed.takeIf { it.exists() && it.isDirectory }
    }

    private fun normalizeSplitDisplayName(requestedSplitName: String, imagesFolder: File): String {
        val parentName = imagesFolder.parentFile?.name?.lowercase(Locale.US)
        val folderName = imagesFolder.name.lowercase(Locale.US)

        // Normalize different aliases into consistent split names
        return when {
            parentName == "valid" || folderName == "valid" || requestedSplitName == "valid" || requestedSplitName == "validation" -> "valid"
            parentName == "val" || folderName == "val" -> "val"
            parentName == "test" || folderName == "test" || requestedSplitName == "testing" -> "test"
            else -> "train"
        }
    }

    private fun parseYoloClassNames(root: File): List<String> {
        val yaml = listOf(
            File(root, "data.yaml"),
            File(root, "dataset.yaml")
        ).firstOrNull { it.exists() }
            ?: root.listFiles()?.firstOrNull {
                it.isFile && (it.name.endsWith(".yaml", true) || it.name.endsWith(".yml", true))
            }
            ?: return emptyList()

        val content = runCatching { yaml.readText() }.getOrNull() ?: return emptyList()
        return parseYamlNames(content)
    }

    private fun parseYamlNames(yamlText: String): List<String> {
        val lines = yamlText.lines()

        // names: [a, b, c]
        val inlineNamesRegex = Regex("""^\s*names\s*:\s*\[(.*)]\s*$""")
        for (line in lines) {
            val match = inlineNamesRegex.find(line)
            if (match != null) {
                return match.groupValues[1]
                    .split(",")
                    .map { it.trim().trim('"', '\'') }
                    .filter { it.isNotEmpty() }
            }
        }

        // names:
        //   0: angry
        //   1: happy
        val blockMapStartIndex = lines.indexOfFirst { it.trim().startsWith("names:") }
        if (blockMapStartIndex >= 0) {
            val namesMap = mutableListOf<String>()
            for (i in (blockMapStartIndex + 1) until lines.size) {
                val line = lines[i]
                if (line.isBlank()) continue
                if (!line.startsWith(" ") && !line.startsWith("\t")) break

                val kvMatch = Regex("""^\s*\d+\s*:\s*(.+?)\s*$""").find(line)
                if (kvMatch != null) {
                    namesMap += kvMatch.groupValues[1].trim().trim('"', '\'')
                }
            }
            if (namesMap.isNotEmpty()) return namesMap
        }

        // names:
        //   - angry
        //   - happy
        if (blockMapStartIndex >= 0) {
            val namesList = mutableListOf<String>()
            for (i in (blockMapStartIndex + 1) until lines.size) {
                val line = lines[i]
                if (line.isBlank()) continue
                if (!line.startsWith(" ") && !line.startsWith("\t")) break

                val dashMatch = Regex("""^\s*-\s*(.+?)\s*$""").find(line)
                if (dashMatch != null) {
                    namesList += dashMatch.groupValues[1].trim().trim('"', '\'')
                }
            }
            if (namesList.isNotEmpty()) return namesList
        }

        return emptyList()
    }

    private fun readYoloGroundTruthClassIds(labelsFolder: File?, imageFile: File): List<Int> {
        if (labelsFolder == null || !labelsFolder.exists() || !labelsFolder.isDirectory) return emptyList()

        val baseName = imageFile.name.substringBeforeLast(".")
        val labelFile = File(labelsFolder, "$baseName.txt")
        if (!labelFile.exists()) return emptyList()

        return runCatching { labelFile.readText() }
            .getOrDefault("")
            .lines()
            .mapNotNull { line ->
                val parts = line.trim().split(Regex("""\s+"""))
                parts.firstOrNull()?.toIntOrNull()
            }
    }

    private fun runSingleYoloInference(bitmap: Bitmap, interpreter: Interpreter): List<YoloDet> {
        // Resize/pad image to the model input size
        val meta = YOLO_LED_Detection.createLetterboxedBitmap(
            srcBitmap = bitmap,
            targetWidth = YOLO_INPUT_W,
            targetHeight = YOLO_INPUT_H
        )

        val inputBuffer = YOLO_LED_Detection.bitmapToNormalizedTensorNHWC(meta.inputBitmap)
        val output = Array(1) { Array(300) { FloatArray(6) } }

        interpreter.run(inputBuffer, output)
        return YOLO_LED_Detection.parseTFLite(output)
    }

    private fun getOrCreateYoloInterpreter(): Interpreter {
        yoloInterpreter?.let { return it }

        val assetPath = findBestYoloTfliteAsset()
            ?: throw IllegalStateException(
                "Could not find a YOLO .tflite asset. Place your YOLO model in assets with a name containing 'yolo' or 'led'."
            )

        val modelBuffer = loadMappedAssetFile(this, assetPath)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }

        return Interpreter(modelBuffer, options).also {
            yoloInterpreter = it
            Log.d(TAG, "Loaded YOLO model from assets: $assetPath")
        }
    }

    private fun findBestYoloTfliteAsset(): String? {
        val allAssets = listAllAssetPaths("")
        return allAssets.firstOrNull {
            it.endsWith(".tflite", true) && (
                    it.contains("yolo", true) ||
                            it.contains("led", true)
                    )
        } ?: allAssets.firstOrNull { it.endsWith(".tflite", true) }
    }

    private fun listAllAssetPaths(path: String): List<String> {
        val children = assets.list(path)?.toList().orEmpty()
        if (children.isEmpty()) {
            return if (path.isNotBlank()) listOf(path) else emptyList()
        }

        val results = mutableListOf<String>()
        for (child in children) {
            val childPath = if (path.isBlank()) child else "$path/$child"
            val grandChildren = assets.list(childPath)?.toList().orEmpty()
            if (grandChildren.isEmpty()) {
                results += childPath
            } else {
                results += listAllAssetPaths(childPath)
            }
        }
        return results
    }

    private fun loadMappedAssetFile(context: Context, assetPath: String): MappedByteBuffer {
        val afd = context.assets.openFd(assetPath)
        FileInputStream(afd.fileDescriptor).channel.use { channel ->
            return channel.map(
                FileChannel.MapMode.READ_ONLY,
                afd.startOffset,
                afd.declaredLength
            )
        }
    }

    // -------------------------------------------------------------------------
    // ResNet dataset inference
    // -------------------------------------------------------------------------

    private suspend fun runResNetInferenceOverDataset(input: PickedInput) {
        val progressUi = createProgressDialog("ResNet-50 Dataset Inference")

        try {
            val result = withContext(Dispatchers.IO) {
                val prepared = prepareInputRoot(input, PICK_MODE_RESNET, progressUi)
                val datasetRoot = prepared.rootDir

                updateProgress(progressUi, 3, 4, "Scanning sequences")

                val sequences = collectResNetSequences(datasetRoot)
                if (sequences.isEmpty()) {
                    throw IllegalStateException(
                        "No valid ResNet sequences were found. Expected folders containing image frames, usually under class/sequence."
                    )
                }

                val classifier = getOrCreatePyTorchClassifier()

                val classNames = sequences.map { it.groundTruthLabel }.distinct().sorted()
                val classIndexMap = classNames.withIndex().associate { it.value to it.index }

                val perClassGt = IntArray(classNames.size)
                val perClassCorrect = IntArray(classNames.size)

                var overallCorrect = 0
                val splitStats = linkedMapOf<String, Pair<Int, Int>>()

                val sb = StringBuilder()
                sb.appendLine("RoEmotion - ResNet-50 Dataset Inference Report")
                sb.appendLine("Generated: ${timestampNow()}")
                sb.appendLine("Selected Input: ${prepared.description}")
                sb.appendLine("Detected Dataset Root: ${datasetRoot.absolutePath}")
                sb.appendLine("Detected Classes: ${classNames.joinToString(", ")}")
                sb.appendLine("Total Sequences: ${sequences.size}")
                sb.appendLine()

                sequences.forEachIndexed { index, seqInfo ->
                    updateProgress(progressUi, index + 1, sequences.size, "ResNet inferring sequences")

                    val frames = loadBitmapSequence(seqInfo.frameFiles)
                    if (frames.isEmpty()) {
                        sb.appendLine("[Skipped] ${seqInfo.relativePath}")
                        sb.appendLine("  Ground Truth: ${seqInfo.groundTruthLabel}")
                        sb.appendLine("  Prediction: [frame load failed]")
                        sb.appendLine()
                        return@forEachIndexed
                    }

                    val preparedFrames = adjustSequenceLength(frames, RESNET_SEQUENCE_LENGTH)
                    val (predLabel, probs) = classifier.classifySequence(preparedFrames)

                    val gtNorm = normalizeLabel(seqInfo.groundTruthLabel)
                    val predNorm = normalizeLabel(predLabel)
                    val success = gtNorm == predNorm

                    classIndexMap[seqInfo.groundTruthLabel]?.let { gtIdx ->
                        perClassGt[gtIdx]++
                        if (success) {
                            perClassCorrect[gtIdx]++
                        }
                    }

                    if (success) overallCorrect++

                    val oldSplit = splitStats[seqInfo.splitName] ?: (0 to 0)
                    splitStats[seqInfo.splitName] = if (success) {
                        (oldSplit.first + 1) to (oldSplit.second + 1)
                    } else {
                        (oldSplit.first + 1) to oldSplit.second
                    }

                    sb.appendLine(seqInfo.relativePath)
                    sb.appendLine("  Split: ${seqInfo.splitName}")
                    sb.appendLine("  Ground Truth: ${seqInfo.groundTruthLabel}")
                    sb.appendLine("  Prediction: $predLabel")
                    sb.appendLine("  Frames Found: ${seqInfo.frameFiles.size}")
                    sb.appendLine("  Frames Used: ${preparedFrames.size}")
                    sb.appendLine("  Success: ${if (success) "Yes" else "No"}")
                    sb.appendLine("  Probabilities: ${formatProbabilities(probs)}")
                    sb.appendLine()
                }

                sb.appendLine("========== Per-Class Summary ==========")
                classNames.forEachIndexed { i, className ->
                    val gt = perClassGt[i]
                    val correct = perClassCorrect[i]
                    val pct = if (gt == 0) 0.0 else (correct * 100.0 / gt.toDouble())
                    sb.appendLine(
                        "$className -> Ground Truth: $gt, Successful: $correct, Percentage: ${
                            String.format(Locale.US, "%.2f", pct)
                        }%"
                    )
                }

                sb.appendLine()
                sb.appendLine("========== Split Summary ==========")
                splitStats.forEach { (split, pair) ->
                    val gt = pair.first
                    val correct = pair.second
                    val pct = if (gt == 0) 0.0 else (correct * 100.0 / gt.toDouble())
                    sb.appendLine(
                        "$split -> Ground Truth: $gt, Successful: $correct, Percentage: ${
                            String.format(Locale.US, "%.2f", pct)
                        }%"
                    )
                }

                val overallPct = if (sequences.isEmpty()) 0.0 else (overallCorrect * 100.0 / sequences.size.toDouble())

                sb.appendLine()
                sb.appendLine("========== Overall Summary ==========")
                sb.appendLine("Overall Ground Truth Count: ${sequences.size}")
                sb.appendLine("Overall Successful Count: $overallCorrect")
                sb.appendLine("Overall Success Percentage: ${String.format(Locale.US, "%.2f", overallPct)}%")

                val fileName = "resnet50_dataset_inference_${fileSafeTimestamp()}.txt"
                val savedUri = saveTextLogToDocuments(fileName, sb.toString())

                InferenceResult(
                    title = "ResNet-50 Inference Finished",
                    fileName = fileName,
                    savedUri = savedUri,
                    reportText = sb.toString()
                )
            }

            closeProgressDialog(progressUi)

            showMessageBox(
                title = result.title,
                message = buildSavedLogSummary(
                    fileName = result.fileName,
                    savedUri = result.savedUri,
                    reportText = result.reportText
                )
            )
        } catch (e: Exception) {
            Log.e(TAG, "ResNet inference failed", e)
            closeProgressDialog(progressUi)
            showMessageBox("ResNet-50 Inference Error", e.message ?: "Unknown error")
        }
    }

    private data class ResNetSequenceInfo(
        val sequenceDir: File,
        val frameFiles: List<File>,
        val groundTruthLabel: String,
        val splitName: String,
        val relativePath: String
    )

    private fun collectResNetSequences(root: File): List<ResNetSequenceInfo> {
        val splitNames = setOf("train", "val", "valid", "validation", "test", "testing")

        val candidateDirs = root.walkTopDown()
            .filter { it.isDirectory }
            .toList()

        val result = mutableListOf<ResNetSequenceInfo>()

        for (dir in candidateDirs) {
            val imageFiles = dir.listFiles()
                ?.filter { it.isFile && isImageFileName(it.name) }
                ?.sortedBy { it.name.lowercase(Locale.US) }
                .orEmpty()

            if (imageFiles.isEmpty()) continue

            val relative = dir.relativeTo(root).invariantSeparatorsPath
            val parts = relative.split("/").filter { it.isNotBlank() }

            // Infer split and class label from directory structure
            val splitName = parts.firstOrNull { it.lowercase(Locale.US) in splitNames } ?: "unknown"
            val gtLabel = inferGroundTruthLabelFromPath(parts, splitNames, dir)

            result += ResNetSequenceInfo(
                sequenceDir = dir,
                frameFiles = imageFiles,
                groundTruthLabel = gtLabel,
                splitName = splitName,
                relativePath = relative
            )
        }

        return result.sortedBy { it.relativePath.lowercase(Locale.US) }
    }

    private fun inferGroundTruthLabelFromPath(
        parts: List<String>,
        splitNames: Set<String>,
        dir: File
    ): String {
        if (parts.isEmpty()) return dir.name

        val splitIndex = parts.indexOfFirst { it.lowercase(Locale.US) in splitNames }
        return when {
            splitIndex >= 0 && parts.size > splitIndex + 1 -> parts[splitIndex + 1]
            parts.size >= 2 -> parts[parts.size - 2]
            else -> parts.first()
        }
    }

    private fun loadBitmapSequence(frameFiles: List<File>): List<Bitmap> {
        return frameFiles.mapNotNull { loadBitmapFromFile(it) }
    }

    private fun adjustSequenceLength(frames: List<Bitmap>, targetLength: Int): List<Bitmap> {
        if (frames.isEmpty()) return emptyList()

        return when {
            frames.size == targetLength -> frames

            frames.size > targetLength -> {
                // Uniformly sample frames if sequence is longer than expected
                val result = mutableListOf<Bitmap>()
                for (i in 0 until targetLength) {
                    val index = ((i.toFloat() / targetLength.toFloat()) * frames.size.toFloat())
                        .toInt()
                        .coerceIn(0, frames.lastIndex)
                    result += frames[index]
                }
                result
            }

            else -> {
                // Pad sequence by repeating the last frame
                val result = frames.toMutableList()
                while (result.size < targetLength) {
                    result += frames.last()
                }
                result
            }
        }
    }

    private fun getOrCreatePyTorchClassifier(): PyTorchClassifier {
        pytorchClassifier?.let { return it }

        return try {
            PyTorchClassifier.fromAsset(
                context = this,
                modelAsset = "RoEmotion_Emotion_Detection_ResNet_50_LSTM_Attention.pt"
            ).also {
                pytorchClassifier = it
            }
        } catch (e: Exception) {
            throw IllegalStateException(
                "Could not load the ResNet-50 PyTorch model from assets. " +
                        "Expected asset: RoEmotion_Emotion_Detection_ResNet_50_LSTM_Attention.pt",
                e
            )
        }
    }

    // -------------------------------------------------------------------------
    // Common helpers
    // -------------------------------------------------------------------------

    private fun loadBitmapFromFile(file: File): Bitmap? {
        return try {
            BitmapFactory.decodeFile(file.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load bitmap: ${file.absolutePath}", e)
            null
        }
    }

    private fun isImageFileName(name: String?): Boolean {
        if (name.isNullOrBlank()) return false
        val lower = name.lowercase(Locale.US)
        return lower.endsWith(".jpg") ||
                lower.endsWith(".jpeg") ||
                lower.endsWith(".png") ||
                lower.endsWith(".bmp") ||
                lower.endsWith(".webp")
    }

    private fun normalizeLabel(label: String): String {
        // Normalize labels for safer comparison
        return label.trim()
            .lowercase(Locale.US)
            .replace("-", " ")
            .replace("_", " ")
            .replace(Regex("""\s+"""), " ")
    }

    private fun formatProbabilities(probs: FloatArray): String {
        return probs.joinToString(", ") { String.format(Locale.US, "%.4f", it) }
    }

    // -------------------------------------------------------------------------
    // Save log to Documents
    // -------------------------------------------------------------------------

    private fun saveTextLogToDocuments(fileName: String, content: String): Uri? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            saveTextLogToDocumentsScoped(fileName, content)
        } else {
            saveTextLogLegacy(fileName, content)
        }
    }

    private fun saveTextLogToDocumentsScoped(fileName: String, content: String): Uri? {
        val resolver = contentResolver
        val collection = MediaStore.Files.getContentUri("external")

        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "text/plain")
            put(
                MediaStore.MediaColumns.RELATIVE_PATH,
                Environment.DIRECTORY_DOCUMENTS + File.separator + "RoEmotion"
            )
        }

        val uri = resolver.insert(collection, values)
            ?: throw IllegalStateException("Failed to create the output log file in Documents.")

        resolver.openOutputStream(uri)?.use { output ->
            output.write(content.toByteArray())
            output.flush()
        } ?: throw IllegalStateException("Failed to write the output log file.")

        return uri
    }

    @Suppress("DEPRECATION")
    private fun saveTextLogLegacy(fileName: String, content: String): Uri? {
        val docsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val outDir = File(docsDir, "RoEmotion")
        if (!outDir.exists()) outDir.mkdirs()

        val outFile = File(outDir, fileName)
        FileOutputStream(outFile).use { it.write(content.toByteArray()) }
        return Uri.fromFile(outFile)
    }

    private fun buildSavedLogSummary(fileName: String, savedUri: Uri?, reportText: String): String {
        val summaryLines = reportText.lines()
            .filter {
                it.startsWith("Overall ") ||
                        it.contains("Percentage:") ||
                        it.startsWith("Success:") ||
                        it.contains("Image-Level Full Match")
            }

        return buildString {
            appendLine("Log file saved.")
            appendLine()
            appendLine("File Name: $fileName")
            if (savedUri != null) {
                appendLine("Saved URI: $savedUri")
            }
            appendLine()
            appendLine("Summary:")
            summaryLines.take(20).forEach { appendLine(it) }
        }.trim()
    }

    private fun timestampNow(): String {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
    }

    private fun fileSafeTimestamp(): String {
        return SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    }

    override fun onDestroy() {
        super.onDestroy()

        // Release model resources
        try {
            yoloInterpreter?.close()
        } catch (_: Exception) {
        }
        yoloInterpreter = null

        try {
            pytorchClassifier?.close()
        } catch (_: Exception) {
        }
        pytorchClassifier = null
    }
}