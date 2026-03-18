package com.developer27.xemotion

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.os.Environment
import android.preference.PreferenceManager
import android.provider.MediaStore
import android.text.InputType
import android.util.Log
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.developer27.xemotion.camera.CameraHelper
import com.developer27.xemotion.databinding.ActivityMainBinding
import com.developer27.xemotion.inference.LocalInferenceActivity
import com.developer27.xemotion.inference.PyTorchClassifier
import com.developer27.xemotion.inference.PyTorchModuleLoader
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoDrawing.Traces.TracePreprocessing
import com.developer27.xemotion.videoprocessing.VideoProcessor
import org.opencv.core.Scalar
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale
import java.util.Timer
import java.util.TimerTask

class MainActivity : AppCompatActivity() {

    // View binding for accessing UI elements
    private lateinit var viewBinding: ActivityMainBinding

    // App preferences storage
    private lateinit var sharedPreferences: SharedPreferences

    // Android camera system manager
    private lateinit var cameraManager: CameraManager

    // Helper class for camera operations (open, close, settings)
    private lateinit var cameraHelper: CameraHelper

    // TFLite YOLO interpreter for LED detection
    private var yoloInterpreter: Interpreter? = null

    // Main video processing pipeline (detection + drawing + labels)
    private var videoProcessor: VideoProcessor? = null

    // Trace export / preprocessing helper
    private val tracePreprocessing = TracePreprocessing()

    // Sequence length used by the emotion model (number of frames)
    private val seqLen = 5

    // Buffers for contour/global sequence inference
    private val traceBuffer = ArrayDeque<Bitmap>(seqLen)
    private val contourTraceBuffer = ArrayDeque<Bitmap>(seqLen)

    // Per-user sequence buffers for YOLO mode (one buffer per detected user)
    private val perUserBuffers = Array(3) { ArrayDeque<Bitmap>(seqLen) }

    // PyTorch-based emotion classifier
    private lateinit var emotionClassifier: PyTorchClassifier

    // Flags for recording and processing states
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false // prevents overlapping frame processing

    // AR mode state and timer
    private var isArMode = false
    private var arTimer: CountDownTimer? = null

    // Timer for periodic export + inference
    private var exportTimer: Timer? = null
    private var batchCount = 0 // counts saved/exported batches

    // UI/control state flags
    private var shouldClearPrediction = false
    private var isFrontCamera = false

    // Cached contour prediction so UI keeps the last result visible
    private var lastContourUserLabel: String = "CONTOUR"
    private var lastContourClassificationLabel: String = ""
    private var lastContourEmotionLabel: String = ""
    private var lastContourConfidencePct: Float = 0f
    private var lastContourColor: Scalar = Scalar(255.0, 255.0, 255.0)

    // Required runtime permissions
    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    // Launcher for requesting camera + audio permissions at runtime
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1 // request code for settings activity

        // Maps device rotation → camera preview orientation
        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }

        private const val TAG = "MainActivity" // log tag
    }

    // TextureView listener for camera preview lifecycle
    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            Log.d(TAG, "onSurfaceTextureAvailable: $width x $height")

            // Open camera if permissions are granted, otherwise request them
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(requiredPermissions)
            }
        }

        // No special handling needed for size changes
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) =
            Unit

        // Return false so system handles surface release
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false

        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            // Process each new frame only when tracking is active
            if (isProcessing) {
                processFrameWithVideoProcessor()
            }
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        // Keep screen awake while app is running
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        installSplashScreen()

        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Initialize core app components
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor(this)

        // Hide processed preview until tracking starts
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.viewFinder.surfaceTextureListener = textureListener

        // Open project website when title is tapped
        viewBinding.titleContainer.setOnClickListener {
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.developer27.com")))
        }

        // Permission launcher for camera + audio
        requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { perms ->
                val camGranted = perms[Manifest.permission.CAMERA] ?: false
                val micGranted = perms[Manifest.permission.RECORD_AUDIO] ?: false

                // Open camera only if both permissions are granted
                if (camGranted && micGranted) {
                    if (viewBinding.viewFinder.isAvailable) {
                        cameraHelper.openCamera()
                    }
                } else {
                    Toast.makeText(
                        this,
                        "Camera & Audio permissions are required.",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

        // Request permissions or open camera immediately
        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            }
        } else {
            requestPermissionLauncher.launch(requiredPermissions)
        }

        // Load PyTorch model once to verify the asset is accessible
        val module = PyTorchModuleLoader.loadFromAssets(
            this,
            "RoEmotion_Emotion_Detection_ResNet_50_LSTM_Attention.pt"
        )
        Log.d(TAG, "Emotion recognition model loaded successfully: $module")

        // Create emotion classifier from PyTorch asset
        emotionClassifier = PyTorchClassifier.fromAsset(
            this,
            "RoEmotion_Emotion_Detection_ResNet_50_LSTM_Attention.pt"
        )

        // Load TFLite detector in the background
        loadTFLiteModelOnStartupThreaded("RoEmotion_LED_Detection_float32.tflite")
        cameraHelper.setupZoomControls()

        // Start or stop tracking
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }

        // Switch between front and back camera
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }

        // Open about screen
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXemotionActivity::class.java))
        }

        // Open settings screen
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // Clear current predictions
        viewBinding.clearPredictionButton.setOnClickListener {
            if (isRecording) stopProcessingAndRecording()
            clearAllPredictions()
        }

        // Open local inference page
        viewBinding.localInferenceButton.setOnClickListener {
            startActivity(Intent(this, LocalInferenceActivity::class.java))
        }

        // Enter or exit AR mode
        viewBinding.arModeButton.setOnClickListener {
            if (isArMode) exitArMode() else promptAndEnterArMode()
        }
    }

    // Ask user how many minutes AR mode should run
    private fun promptAndEnterArMode() {
        val margin = (16 * resources.displayMetrics.density).toInt()

        // Numeric input for AR session duration
        val input = EditText(this).apply {
            hint = "Minutes"
            inputType = InputType.TYPE_CLASS_NUMBER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                setMargins(margin, margin, margin, 0)
            }
        }

        // Show duration dialog, then convert minutes -> milliseconds
        AlertDialog.Builder(this)
            .setTitle("AR Session Duration")
            .setMessage("Enter how many minutes to run AR mode:")
            .setView(input)
            .setPositiveButton("Start") { _, _ ->
                val minutes = input.text.toString().toLongOrNull()?.coerceAtLeast(1) ?: 1L
                enterArMode(minutes * 60_000L)
            }
            .setNegativeButton("Cancel", null)
            .setCancelable(true)
            .show()
    }

    // Enter AR mode, hide system UI, and force AR rolling shutter settings
    private fun enterArMode(durationMs: Long) {
        isArMode = true
        updateArButtonUi(true)
        hideUiForAr(true)

        // Hide system bars for immersive AR mode
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            window.insetsController?.let { ic ->
                ic.hide(WindowInsets.Type.systemBars())
                ic.systemBarsBehavior =
                    WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility =
                (View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        or View.SYSTEM_UI_FLAG_FULLSCREEN
                        or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY)
        }

        // Start tracking automatically if not already running
        if (!isRecording) startProcessingAndRecording()

        // Force AR-specific rolling shutter camera settings
        cameraHelper.forceArRollingShutter()

        // Restart AR countdown timer
        arTimer?.cancel()
        arTimer = object : CountDownTimer(durationMs, 1_000L) {
            override fun onTick(millisUntilFinished: Long) = Unit

            override fun onFinish() {
                exitArMode()
            }
        }.start()
    }

    // Exit AR mode and restore normal UI/camera behavior
    private fun exitArMode() {
        if (!isArMode) return

        isArMode = false
        arTimer?.cancel()
        arTimer = null

        updateArButtonUi(false)
        hideUiForAr(false)

        // Stop active processing/export loop
        if (isRecording) {
            isRecording = false
            isProcessing = false
            exportTimer?.cancel()
            exportTimer = null
        }

        // Restore normal system UI
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(true)
            window.insetsController?.show(WindowInsets.Type.systemBars())
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_VISIBLE
        }

        // Restore normal camera shutter settings
        cameraHelper.updateShutterSpeed()

        // Reset processing UI
        viewBinding.startProcessingButton.text = getString(R.string.start_capture)
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
    }

    // Update AR button state
    private fun updateArButtonUi(ar: Boolean) {
        viewBinding.arModeButton.apply {
            // Toggle AR button label and color
            text = if (ar) "Exit AR" else "AR Mode"
            backgroundTintList = ColorStateList.valueOf(
                ContextCompat.getColor(this@MainActivity, if (ar) R.color.red else R.color.green)
            )
            setTextColor(ContextCompat.getColor(this@MainActivity, android.R.color.white))
        }
    }

    // Hide normal UI while AR mode is active
    private fun hideUiForAr(hide: Boolean) {
        // Hide non-AR controls, keep processed frame visible only when processing
        viewBinding.titleContainer.visibility = if (hide) View.GONE else View.VISIBLE
        viewBinding.topControlsContainer.visibility = if (hide) View.GONE else View.VISIBLE
        viewBinding.zoomButtonContainer.visibility = if (hide) View.GONE else View.VISIBLE
        viewBinding.buttonContainer.visibility = if (hide) View.GONE else View.VISIBLE
        viewBinding.processedFrameView.visibility = if (isProcessing) View.VISIBLE else View.GONE
    }

    // Restore auto exposure if manual exposure was previously forced
    private fun trySetAutoExposure() {
        val builder = cameraHelper.captureRequestBuilder ?: return

        // Switch camera back to automatic exposure mode
        builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
        builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)

        try {
            cameraHelper.cameraCaptureSession?.setRepeatingRequest(
                builder.build(),
                null,
                cameraHelper.backgroundHandler
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to revert AE: ${e.message}")
        }
    }

    // Start frame processing, exporting, and timed inference ticks
    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true

        // Update UI to tracking state
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        // Reset counters, buffers, and processor state
        batchCount = 0
        clearAllBuffers()

        videoProcessor?.reset()
        clearAllPredictions()

        // Use different periodic export intervals for contour vs YOLO
        val intervalMs = when (Settings.DetectionMode.current) {
            Settings.DetectionMode.Mode.CONTOUR -> 700L
            Settings.DetectionMode.Mode.YOLO -> 800L
        }

        exportTimer?.cancel()
        exportTimer = Timer()
        exportTimer?.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                runOnUiThread {
                    val vp = videoProcessor ?: return@runOnUiThread

                    // Export traces and run inference based on current detection mode
                    when (Settings.DetectionMode.current) {
                        Settings.DetectionMode.Mode.YOLO -> processYoloExportTick(vp)
                        Settings.DetectionMode.Mode.CONTOUR -> processContourExportTick(vp)
                    }

                    appendPredictionToLog("log")

                    // Reset per-tick processor state
                    vp.reset()

                    // Re-apply cached contour prediction after reset
                    if (Settings.DetectionMode.current == Settings.DetectionMode.Mode.CONTOUR) {
                        restoreContourPredictionState()
                    }
                }
            }
        }, intervalMs, intervalMs)
    }

    // Export per-user traces and run user-specific inference in YOLO mode
    private fun processYoloExportTick(vp: VideoProcessor) {
        val order = vp.getDetectionOrder()
        if (order.isEmpty()) return

        for (cid in order) {
            if (cid !in perUserBuffers.indices) continue

            // Save all trace variants for this detected user
            tracePreprocessing.exportPerUserTraceVariantsSnapshot(
                classId = cid,
                perClassRaw = vp.getPerClassRaw()
            )?.let { variants ->
                savePerUserBitmap(variants.raw, cid, "raw")
                savePerUserBitmap(variants.rawCv, cid, "rawCv")
                savePerUserBitmap(variants.spline, cid, "spline")
                savePerUserBitmap(variants.splineCv, cid, "splineCv")
            }

            // Export processed trace used for sequence inference
            val userBmp = tracePreprocessing.exportProcessedTraceForClass(
                classId = cid,
                perClassRaw = vp.getPerClassRaw(),
                perClassSmooth = vp.getPerClassSmooth()
            ) ?: continue

            val buf = perUserBuffers[cid]
            buf.addLast(userBmp)
            if (buf.size > seqLen) buf.removeFirst()

            // Run inference once enough frames are collected
            if (buf.size == seqLen) {
                runPerUserInferenceAndLabel(cid, buf.toList())
                buf.clear()
            }
        }
    }

    // Export contour trace and run contour-based inference
    private fun processContourExportTick(vp: VideoProcessor) {
        // Build processed contour trace from current smooth points
        val contourBmp = tracePreprocessing.exportSplineTraceWithCvProcessing(
            vp.getSmoothDataList()
        )

        // Optionally save exported contour image
        if (Settings.ExportData.frameIMG) {
            saveContourBitmap(contourBmp)
        }

        // Keep only the latest seqLen contour frames
        contourTraceBuffer.addLast(contourBmp)
        if (contourTraceBuffer.size > seqLen) {
            contourTraceBuffer.removeFirst()
        }

        if (contourTraceBuffer.size == seqLen) {
            // Run sequence inference when enough contour frames exist
            runContourInferenceAndLabel(contourTraceBuffer.toList())
            contourTraceBuffer.clear()
        } else {
            // Keep previous contour result visible until next inference is ready
            restoreContourPredictionState()
        }
    }

    // Stop processing and run last export/inference pass if needed
    private fun stopProcessingAndRecording() {
        if (!isRecording) return

        isRecording = false
        isProcessing = false

        // Stop periodic export/inference timer
        exportTimer?.cancel()
        exportTimer = null

        try {
            val vp = videoProcessor
            if (vp != null) {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR -> {
                        // Final contour export + inference before stopping
                        val traceBitmap = tracePreprocessing.exportSplineTraceWithCvProcessing(
                            vp.getSmoothDataList()
                        )
                        saveBatchAndRunInference(traceBitmap)
                        restoreContourPredictionState()
                    }

                    Settings.DetectionMode.Mode.YOLO -> {
                        // Final per-user export + inference before stopping
                        for (cid in vp.getDetectionOrder()) {
                            if (cid !in perUserBuffers.indices) continue

                            val userBmp = tracePreprocessing.exportProcessedTraceForClass(
                                classId = cid,
                                perClassRaw = vp.getPerClassRaw(),
                                perClassSmooth = vp.getPerClassSmooth()
                            ) ?: continue

                            val buf = perUserBuffers[cid]
                            buf.addLast(userBmp)

                            while (buf.size > seqLen) {
                                buf.removeFirst()
                            }

                            if (buf.size >= seqLen) {
                                runPerUserInferenceAndLabel(cid, buf.toList())
                                buf.clear()
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting trace", e)
        }

        // Show export summary if image saving is enabled
        if (Settings.ExportData.frameIMG) {
            Toast.makeText(this, "$batchCount batches have been saved", Toast.LENGTH_LONG).show()
        }

        clearAllBuffers()

        // Reset UI back to idle state
        viewBinding.startProcessingButton.text = getString(R.string.start_capture)
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        videoProcessor?.classificationLabel = ""
    }

    // Run sequence classification for one detected user
    private fun runPerUserInferenceAndLabel(classId: Int, frames: List<Bitmap>) {
        // Classify the sequence of frames for this user
        val (bestLabel, probs) = emotionClassifier.classifySequence(frames)
        val bestIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val confidencePct = probs[bestIdx] * 100f

        emotionClassifier.classifyAndLogSequence(frames) // optional detailed logging

        // Resolve user label (fallback if processor not ready)
        val userName = videoProcessor?.labelForClass(classId) ?: "User_${classId + 1}"

        // Build display text: User | Emotion (Confidence)
        val textResult = String.format(
            Locale.US,
            "%s | %s (%.2f%%)",
            userName,
            bestLabel,
            confidencePct
        )

        // Select display color based on predicted emotion
        val textColorScalar = colorForEmotion(bestLabel)

        // Store prediction for drawing (box + label)
        videoProcessor?.setPerUserEmotion(classId, bestLabel, textColorScalar, confidencePct)

        // Update global labels shown on UI
        videoProcessor?.userLabel = userName
        videoProcessor?.classificationLabel = textResult
    }

    // Run sequence classification for contour mode
    private fun runContourInferenceAndLabel(frames: List<Bitmap>) {
        // Classify the current contour sequence
        val (bestLabel, probs) = emotionClassifier.classifySequence(frames)
        val bestIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val confidencePct = probs[bestIdx] * 100f

        emotionClassifier.classifyAndLogSequence(frames) // optional detailed logging

        val contourLabel = "CONTOUR"
        val emotionText = String.format(
            Locale.US,
            "%s (%.2f%%)",
            bestLabel,
            confidencePct
        )

        // Pick display color based on predicted emotion
        val textColorScalar = colorForEmotion(bestLabel)

        // Cache contour prediction so it can be restored later
        lastContourUserLabel = contourLabel
        lastContourEmotionLabel = bestLabel
        lastContourConfidencePct = confidencePct
        lastContourClassificationLabel = emotionText
        lastContourColor = textColorScalar

        // Update current labels shown by the video processor
        videoProcessor?.userLabel = contourLabel
        videoProcessor?.classificationLabel = emotionText

        try {
            // Store contour emotion as user 0 for drawing logic
            videoProcessor?.setPerUserEmotion(0, bestLabel, textColorScalar, confidencePct)
        } catch (e: Exception) {
            Log.w(TAG, "Contour per-user emotion update skipped: ${e.message}")
        }
    }

    // Restore cached contour labels after processor resets
    private fun restoreContourPredictionState() {
        // Only restore labels in contour mode
        if (Settings.DetectionMode.current != Settings.DetectionMode.Mode.CONTOUR) return

        // Nothing to restore if no cached contour prediction exists
        if (lastContourClassificationLabel.isBlank()) return

        // Restore cached labels back into the video processor
        videoProcessor?.userLabel = lastContourUserLabel
        videoProcessor?.classificationLabel = lastContourClassificationLabel

        try {
            // Restore cached contour emotion/color/confidence for drawing
            videoProcessor?.setPerUserEmotion(
                0,
                lastContourEmotionLabel,
                lastContourColor,
                lastContourConfidencePct
            )
        } catch (e: Exception) {
            Log.w(TAG, "Contour restore per-user emotion skipped: ${e.message}")
        }
    }

    // Save final contour batch and run inference when stopping
    private fun saveBatchAndRunInference(traceBitmap: Bitmap) {
        // Skip saving/inference if export is disabled
        if (!Settings.ExportData.frameIMG) return

        batchCount++

        // Keep only the latest seqLen contour frames
        traceBuffer.addLast(traceBitmap)
        if (traceBuffer.size > seqLen) {
            traceBuffer.removeFirst()
        }

        // Wait until enough frames are collected for sequence inference
        if (traceBuffer.size < seqLen) return

        val frames = traceBuffer.toList()
        val (bestLabel, probs) = emotionClassifier.classifySequence(frames)
        emotionClassifier.classifyAndLogSequence(frames) // optional detailed logging

        // Convert top probability to percentage
        val confidencePct = (probs.maxOrNull() ?: 0f) * 100f
        val contourLabel = "CONTOUR"
        val emotionText = String.format(
            Locale.US,
            "%s (%.2f%%)",
            bestLabel,
            confidencePct
        )

        // Cache contour prediction so UI can keep showing it
        lastContourUserLabel = contourLabel
        lastContourEmotionLabel = bestLabel
        lastContourConfidencePct = confidencePct
        lastContourClassificationLabel = emotionText
        lastContourColor = colorForEmotion(bestLabel)

        // Push latest contour prediction into the video processor
        videoProcessor?.userLabel = contourLabel
        videoProcessor?.classificationLabel = emotionText

        traceBuffer.clear() // reset sequence buffer after inference
    }

    // Save one per-user bitmap into a user-specific subfolder
    private fun savePerUserBitmap(bmp: Bitmap, classId: Int, subfolder: String) {
        if (!Settings.ExportData.frameIMG) return

        @Suppress("DEPRECATION")
        val picturesDir =
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)

        val label = videoProcessor?.labelForClass(classId) ?: "User_${classId + 1}"
        val baseDir = File(picturesDir, "Exported Lines from Xemotion/$label")
        val dir = File(baseDir, subfolder).apply {
            if (!exists()) mkdirs()
        }

        val filename = "Line_${label}_${batchCount + 1}.jpg"
        val outFile = File(dir, filename)

        try {
            FileOutputStream(outFile).use { out ->
                bmp.compress(Bitmap.CompressFormat.JPEG, 90, out)
                out.flush()
            }
            Log.d(TAG, "Saved: ${outFile.absolutePath}")
            batchCount++
        } catch (e: IOException) {
            Log.w(TAG, "Fallback to MediaStore", e)
            saveViaMediaStore(bmp)
        }
    }

    // Save contour bitmap into the contour export folder
    private fun saveContourBitmap(bmp: Bitmap) {
        if (!Settings.ExportData.frameIMG) return

        @Suppress("DEPRECATION")
        val picturesDir =
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)

        val baseDir = File(picturesDir, "Exported Lines from Xemotion/Contour")
        if (!baseDir.exists()) baseDir.mkdirs()

        val filename = "Line_Contour_${batchCount + 1}.jpg"
        val outFile = File(baseDir, filename)

        try {
            FileOutputStream(outFile).use { out ->
                bmp.compress(Bitmap.CompressFormat.JPEG, 90, out)
                out.flush()
            }
            Log.d(TAG, "Saved contour: ${outFile.absolutePath}")
            batchCount++
        } catch (e: IOException) {
            Log.w(TAG, "Contour fallback to MediaStore", e)
            saveViaMediaStore(bmp)
        }
    }

    // Fallback image save path through MediaStore
    private fun saveViaMediaStore(traceBitmap: Bitmap) {
        // Determine filename based on current user label
        val userName = videoProcessor?.userLabel?.takeIf { it.isNotBlank() } ?: "User_1"
        val filename = "Line_${batchCount + 1}_$userName.jpg"

        // Prepare metadata for MediaStore entry
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                Environment.DIRECTORY_PICTURES + "/Exported Lines from Xemotion"
            )
            put(MediaStore.Images.Media.IS_PENDING, 1) // mark as writing in progress
        }

        val resolver = contentResolver
        val uri = resolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            values
        ) ?: run {
            Log.e(TAG, "MediaStore insert failed") // insert failed
            return
        }

        try {
            // Write bitmap data to MediaStore
            resolver.openOutputStream(uri)?.use { out ->
                traceBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }

            // Mark file as complete
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)

            batchCount++
            Log.d(TAG, "Batch #$batchCount saved via MediaStore: $uri")
        } catch (e: IOException) {
            Log.e(TAG, "Fallback save via MediaStore failed", e) // write error
        }
    }

    // Append latest prediction to a text log if enabled
    private fun appendPredictionToLog(fullLabel: String) {
        if (!Settings.ExportData.enablePredictionLogging) return

        val label = videoProcessor?.classificationLabel ?: ""
        val ts = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
        val line = "$ts => $label"

        @Suppress("DEPRECATION")
        val docDir =
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        if (!docDir.exists()) docDir.mkdirs()

        val logFile = File(docDir, "RoEmotion_Predictions_Log.txt")

        try {
            logFile.appendText("$line\n")
        } catch (e: IOException) {
            Log.e(TAG, "Error writing prediction log: ${e.message}")
        }
    }

    // Process the current preview frame through VideoProcessor
    private fun processFrameWithVideoProcessor() {
        // Prevent overlapping frame processing
        if (isProcessingFrame) return

        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true

        // Restore last contour prediction before processing the next contour frame
        if (Settings.DetectionMode.current == Settings.DetectionMode.Mode.CONTOUR) {
            restoreContourPredictionState()
        }

        // Process current frame and update preview output
        videoProcessor?.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                processedFrames?.let { (outputBitmap, _) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                    }
                }
                isProcessingFrame = false // unlock next frame
            }
        } ?: run {
            isProcessingFrame = false // reset if processor is null
        }
    }

    // Load the detector model asynchronously and attach delegates if available
    private fun loadTFLiteModelOnStartupThreaded(modelName: String) {
        Thread {
            // Copy model from assets → internal storage (background thread)
            val bestLoadedPath = copyAssetModelBlocking(modelName)

            runOnUiThread {
                // If loading failed, notify user and exit
                if (bestLoadedPath.isEmpty()) {
                    Toast.makeText(this, "Failed to load $modelName", Toast.LENGTH_SHORT).show()
                    return@runOnUiThread
                }

                try {
                    // Configure interpreter (use all available CPU cores)
                    val options = Interpreter.Options().apply {
                        setNumThreads(Runtime.getRuntime().availableProcessors())
                    }

                    var delegateAdded = false

                    try {
                        // Try NNAPI first (hardware acceleration on supported devices)
                        val nnApiDelegate = NnApiDelegate()
                        options.addDelegate(nnApiDelegate)
                        delegateAdded = true
                        Log.d(TAG, "NNAPI delegate added.")
                    } catch (e: Exception) {
                        Log.d(TAG, "NNAPI delegate unavailable, fallback to GPU", e)
                    }

                    if (!delegateAdded) {
                        try {
                            // Fallback to GPU delegate if NNAPI is not available
                            val gpuDelegate = GpuDelegate()
                            options.addDelegate(gpuDelegate)
                            Log.d(TAG, "GPU delegate added.")
                        } catch (e: Exception) {
                            // Final fallback: CPU only
                            Log.d(TAG, "GPU delegate unavailable, CPU only.", e)
                        }
                    }

                    // Initialize YOLO interpreter and attach to video processor
                    if (modelName == "RoEmotion_LED_Detection_float32.tflite") {
                        yoloInterpreter = Interpreter(loadMappedFile(bestLoadedPath), options)
                        yoloInterpreter?.let { videoProcessor?.setInterpreter(it) }
                    }
                } catch (e: Exception) {
                    // Handle interpreter creation errors
                    Toast.makeText(
                        this,
                        "Error loading TFLite model: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    Log.e(TAG, "TFLite Interpreter error", e)
                }
            }
        }.start()
    }

    // Memory-map the copied model file for efficient reading
    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        FileInputStream(File(modelPath)).use { fis ->
            val channel = fis.channel
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        }
    }

    // Copy model asset into app files directory if not already present
    private fun copyAssetModelBlocking(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)

            // Reuse existing file if already copied and valid
            if (outFile.exists() && outFile.length() > 0) {
                outFile.absolutePath
            } else {
                // Copy model from assets → internal storage
                assets.open(assetName).use { input ->
                    FileOutputStream(outFile).use { output ->
                        val buffer = ByteArray(4 * 1024) // chunked copy buffer
                        var bytesRead: Int
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                        }
                        output.flush() // ensure all bytes are written
                    }
                }
                outFile.absolutePath
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error copying asset $assetName: ${e.message}") // log failure
            ""
        }
    }

    // Toggle between front and back camera
    private fun switchCamera() {
        if (isRecording) stopProcessingAndRecording()

        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }

    // Take actions when you get back to the app
    override fun onResume() {
        super.onResume()

        cameraHelper.startBackgroundThread()

        if (viewBinding.viewFinder.isAvailable && allPermissionsGranted()) {
            cameraHelper.openCamera()
        }

        if (shouldClearPrediction) {
            clearAllPredictions()
            shouldClearPrediction = false
        }
    }

    // Take actions when another app was opened
    override fun onPause() {
        if (isRecording) {
            stopProcessingAndRecording()
        }

        arTimer?.cancel()
        arTimer = null

        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()

        super.onPause()
    }

    // Check whether all required permissions are granted
    private fun allPermissionsGranted(): Boolean {
        return requiredPermissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    // Clear all sequence buffers
    private fun clearAllBuffers() {
        traceBuffer.clear()
        contourTraceBuffer.clear()
        for (buf in perUserBuffers) {
            buf.clear()
        }
    }

    // Clear active labels and cached contour labels
    private fun clearAllPredictions() {
        videoProcessor?.classificationLabel = ""
        videoProcessor?.userLabel = ""

        lastContourUserLabel = "CONTOUR"
        lastContourClassificationLabel = ""
        lastContourEmotionLabel = ""
        lastContourConfidencePct = 0f
        lastContourColor = Scalar(255.0, 255.0, 255.0)
    }

    // Map emotion labels to display colors
    private fun colorForEmotion(label: String): Scalar {
        return when {
            label.contains("Angry", true) -> Scalar(255.0, 102.0, 102.0)
            label.contains("Anxious", true) -> Scalar(255.0, 255.0, 153.0)
            label.contains("Disgust", true) -> Scalar(153.0, 255.0, 153.0)
            label.contains("Excited", true) -> Scalar(255.0, 204.0, 153.0)
            label.contains("Sad", true) -> Scalar(153.0, 204.0, 255.0)
            else -> Scalar(255.0, 255.0, 255.0)
        }
    }
}