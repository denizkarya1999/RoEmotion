package com.developer27.xemotion

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.CountDownTimer
import android.text.InputType
import android.util.Log
import android.view.TextureView
import android.view.View
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
import androidx.preference.PreferenceManager
import com.developer27.xemotion.ar.ArModeManager
import com.developer27.xemotion.camera.CameraHelper
import com.developer27.xemotion.databinding.ActivityMainBinding
import com.developer27.xemotion.inference.EmotionInference
import com.developer27.xemotion.inference.LocalInferenceActivity
import com.developer27.xemotion.ui.applySystemBarMargins
import com.developer27.xemotion.ui.applySystemBarPadding
import com.developer27.xemotion.ui.enableRoEmotionEdgeToEdge
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoProcessor
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    // View binding for accessing UI elements
    private lateinit var viewBinding: ActivityMainBinding

    // App preferences storage
    private lateinit var sharedPreferences: SharedPreferences

    // Helper class for camera operations (open, close, settings)
    private lateinit var cameraHelper: CameraHelper

    // Main video processing pipeline (detection + drawing + labels)
    private var videoProcessor: VideoProcessor? = null

    // Activity-facing inference coordinator
    private lateinit var emotionInference: EmotionInference

    // AR Mode manager
    private lateinit var arModeManager: ArModeManager
    private var configuredOperatingMode: Settings.OperatingMode.Mode? = null

    // Flags for recording and processing states
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false // prevents overlapping frame processing
    private var isStopping = false
    private var isCountingDown = false
    private var isActivityResumed = false
    private var isPermissionRequestInFlight = false
    private var collectionDialog: AlertDialog? = null
    private var startCountdown: CountDownTimer? = null
    private val afterStopActions = mutableListOf<() -> Unit>()

    // UI/control state flags
    private var isFrontCamera = false
    private var displayedProcessedBitmap: Bitmap? = null

    // Required runtime permissions
    private val requiredPermissions = buildList {
        add(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
    }.toTypedArray()

    // Launcher for requesting camera + audio permissions at runtime
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        private const val TAG = "MainActivity" // log tag
        private const val COUNTDOWN_DURATION_MILLIS = 5_000L
        private const val MAX_PROCESSING_FRAME_EDGE = 1_280
    }

    // TextureView listener for camera preview lifecycle
    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            Log.d(TAG, "onSurfaceTextureAvailable: $width x $height")
            if (!isActivityResumed) return
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestRequiredPermissions()
            }
        }

        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
            cameraHelper.configurePreviewTransform(width, height)
        }

        // Stop using the surface and let TextureView release its SurfaceTexture.
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            cameraHelper.closeCamera()
            return true
        }

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
        installSplashScreen()

        super.onCreate(savedInstanceState)
        enableRoEmotionEdgeToEdge()
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        viewBinding.titleContainer.applySystemBarPadding(bottom = false)
        viewBinding.topControlsContainer.applySystemBarPadding(top = false, bottom = false)
        viewBinding.zoomButtonContainer.applySystemBarMargins(end = true)
        viewBinding.buttonContainer.applySystemBarMargins(bottom = true)

        // Initialize core app components
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        Settings.load(sharedPreferences)
        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor()
        emotionInference = EmotionInference(this, videoProcessor)

        // Hide processed preview until tracking starts
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.countdownText.accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_ASSERTIVE
        viewBinding.viewFinder.surfaceTextureListener = textureListener

        // Open project website when title is tapped
        viewBinding.titleContainer.setOnClickListener {
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.developer27.com")))
        }

        // Permission launcher for camera + audio
        requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
                isPermissionRequestInFlight = false
                if (allPermissionsGranted()) {
                    if (isActivityResumed && viewBinding.viewFinder.isAvailable) {
                        cameraHelper.openCamera()
                    }
                } else {
                    Toast.makeText(
                        this,
                        "Camera and required storage permissions must be granted.",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }

        if (!allPermissionsGranted()) requestRequiredPermissions()

        arModeManager = ArModeManager(
            this,
            viewBinding,
            cameraHelper,
            { isRecording || isCountingDown },
            { startProcessingAndRecording() },
            {
                stopProcessingAndRecording()
                updateOperatingModeUi()
            }
        )
        applyOperatingModeConfiguration()

        cameraHelper.setupZoomControls()

        // Start or stop tracking
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording || isCountingDown) {
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
            if (isRecording || isStopping) {
                stopProcessingAndRecording(emotionInference::clearAllPredictions)
            } else {
                emotionInference.clearAllPredictions()
            }
        }

        // Open local inference page
        viewBinding.localInferenceButton.setOnClickListener {
            startActivity(Intent(this, LocalInferenceActivity::class.java))
        }

        // Enter or exit AR mode
        viewBinding.arModeButton.setOnClickListener {
            if (Settings.OperatingMode.current != Settings.OperatingMode.Mode.INFERENCE) return@setOnClickListener
            if (!emotionInference.areModelsLoaded()) {
                ensureInferenceModelsLoaded()
                Toast.makeText(this, "Inference models are still loading.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (arModeManager.isArMode) arModeManager.exitArMode() else arModeManager.promptAndEnterArMode()
        }
    }

    private fun applyOperatingModeConfiguration() {
        Settings.load(sharedPreferences)
        val mode = Settings.OperatingMode.current
        if (configuredOperatingMode != mode && isCountingDown) cancelStartCountdown()
        if (isStopping) {
            updateOperatingModeUi()
            return
        }
        if (configuredOperatingMode != mode) {
            if (isRecording) {
                stopProcessingAndRecording()
                return
            }
            if (arModeManager.isArMode) arModeManager.exitArMode()
            if (isStopping) return
            when (mode) {
                Settings.OperatingMode.Mode.DATA_COLLECTION -> emotionInference.unloadModels()
                Settings.OperatingMode.Mode.INFERENCE -> ensureInferenceModelsLoaded()
            }
            configuredOperatingMode = mode
        } else if (mode == Settings.OperatingMode.Mode.INFERENCE) {
            ensureInferenceModelsLoaded()
        }
        updateOperatingModeUi()
    }

    private fun ensureInferenceModelsLoaded() {
        emotionInference.loadModels { result ->
            result.onFailure { error ->
                Log.e(TAG, "Inference model startup failed", error)
                Toast.makeText(
                    this,
                    "Unable to load inference models: ${error.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun updateOperatingModeUi() {
        val capabilities = Settings.capabilitiesFor(Settings.OperatingMode.current)
        val inferenceActionsEnabled = capabilities.inferenceActionsEnabled
        val auxiliaryControlsEnabled = !isStopping && !isCountingDown
        viewBinding.titleText.text = getString(R.string.app_name)
        listOf(
            viewBinding.arModeButton,
            viewBinding.localInferenceButton,
            viewBinding.clearPredictionButton
        ).forEach { action ->
            action.isEnabled = inferenceActionsEnabled && auxiliaryControlsEnabled
            action.visibility = if (inferenceActionsEnabled) View.VISIBLE else View.GONE
        }
        listOf(
            viewBinding.switchCameraButton,
            viewBinding.settingsButton,
            viewBinding.aboutButton,
            viewBinding.zoomInButton,
            viewBinding.zoomOutButton,
            viewBinding.titleContainer
        ).forEach { control -> control.isEnabled = auxiliaryControlsEnabled }
        viewBinding.startProcessingButton.isEnabled = !isStopping
        if (isRecording || isCountingDown) {
            viewBinding.startProcessingButton.text = getString(R.string.stop_processing)
            viewBinding.startProcessingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.red)
        } else {
            viewBinding.startProcessingButton.text = getString(R.string.start_processing)
            viewBinding.startProcessingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.blue)
        }
    }

    // Start camera-frame processing and the independent inference session
    private fun startProcessingAndRecording() {
        if (isStopping || isCountingDown) return
        if (videoProcessor?.isOpenCvReady != true) {
            Toast.makeText(this, R.string.opencv_initialization_failed, Toast.LENGTH_LONG).show()
            return
        }
        if (Settings.OperatingMode.current == Settings.OperatingMode.Mode.DATA_COLLECTION) {
            promptForCollectionDetails()
            return
        }
        if (!emotionInference.areModelsLoaded()) {
            ensureInferenceModelsLoaded()
            Toast.makeText(this, "Inference models are still loading.", Toast.LENGTH_SHORT).show()
            return
        }
        beginStartCountdown(collectionUserName = null, collectionEmotion = null)
    }

    private fun promptForCollectionDetails() {
        if (collectionDialog?.isShowing == true) return
        val userNameInput = EditText(this).apply {
            hint = getString(R.string.collection_user_name_hint)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            isSingleLine = true
        }
        val emotionInput = EditText(this).apply {
            hint = getString(R.string.collection_emotion_hint)
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            isSingleLine = true
        }
        val horizontalPadding = (24 * resources.displayMetrics.density).toInt()
        val inputContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(horizontalPadding, 0, horizontalPadding, 0)
            addView(userNameInput)
            addView(emotionInput)
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.data_collection_title)
            .setMessage(R.string.collection_details_question)
            .setView(inputContainer)
            .setPositiveButton(R.string.start_processing, null)
            .setNegativeButton(android.R.string.cancel, null)
            .create()
        collectionDialog = dialog
        dialog.setOnDismissListener {
            if (collectionDialog === dialog) collectionDialog = null
        }

        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val userName = userNameInput.text.toString().trim()
                val emotion = emotionInput.text.toString().trim()
                if (userName.isEmpty()) {
                    userNameInput.error = getString(R.string.collection_user_name_required)
                }
                if (emotion.isEmpty()) {
                    emotionInput.error = getString(R.string.collection_emotion_required)
                }
                if (userName.isNotEmpty() && emotion.isNotEmpty()) {
                    dialog.dismiss()
                    beginStartCountdown(userName, emotion)
                }
            }
            userNameInput.requestFocus()
        }
        dialog.show()
    }

    private fun beginStartCountdown(
        collectionUserName: String?,
        collectionEmotion: String?
    ) {
        if (isRecording || isStopping || isCountingDown) return
        val requestedMode = Settings.OperatingMode.current
        isCountingDown = true
        viewBinding.countdownText.visibility = View.VISIBLE
        updateOperatingModeUi()

        startCountdown = object : CountDownTimer(COUNTDOWN_DURATION_MILLIS, 1_000L) {
            override fun onTick(millisUntilFinished: Long) {
                val value = ((millisUntilFinished + 999L) / 1_000L).coerceIn(1L, 3L)
                val text = value.toString()
                viewBinding.countdownText.text = text
            }

            override fun onFinish() {
                startCountdown = null
                isCountingDown = false
                hideCountdownOverlay()
                updateOperatingModeUi()
                if (!isActivityResumed || Settings.OperatingMode.current != requestedMode) return
                beginProcessingSession(collectionUserName, collectionEmotion)
            }
        }.start()
    }

    private fun cancelStartCountdown() {
        if (!isCountingDown && startCountdown == null) return
        startCountdown?.cancel()
        startCountdown = null
        isCountingDown = false
        hideCountdownOverlay()
        updateOperatingModeUi()
    }

    private fun hideCountdownOverlay() {
        viewBinding.countdownText.text = ""
        viewBinding.countdownText.visibility = View.GONE
    }

    private fun beginProcessingSession(
        collectionUserName: String?,
        collectionEmotion: String?
    ) {
        if (isStopping) return
        if (Settings.OperatingMode.current == Settings.OperatingMode.Mode.INFERENCE &&
            !emotionInference.areModelsLoaded()
        ) {
            ensureInferenceModelsLoaded()
            Toast.makeText(this, "Inference models are still loading.", Toast.LENGTH_SHORT).show()
            return
        }
        isRecording = true
        isProcessing = true

        // Update UI to tracking state
        viewBinding.startProcessingButton.text = getString(R.string.stop_processing)
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        emotionInference.startSession(collectionUserName, collectionEmotion)
    }

    // Stop processing and run last export/inference pass if needed
    private fun stopProcessingAndRecording(afterStopped: (() -> Unit)? = null) {
        afterStopped?.let(afterStopActions::add)
        if (isStopping) return
        if (isCountingDown) {
            cancelStartCountdown()
            runAfterStopActions()
            return
        }
        if (!isRecording) {
            runAfterStopActions()
            return
        }

        isRecording = false
        isProcessing = false
        isStopping = true
        viewBinding.startProcessingButton.isEnabled = false

        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        displayedProcessedBitmap?.recycle()
        displayedProcessedBitmap = null
        updateOperatingModeUi()

        emotionInference.stopSession { savedTraceCount ->
            isStopping = false
            if (isDestroyed) return@stopSession
            savedTraceCount?.let { count ->
                Toast.makeText(
                    this,
                    resources.getQuantityString(R.plurals.saved_trace_count, count, count),
                    Toast.LENGTH_LONG
                ).show()
            }
            updateOperatingModeUi()
            runAfterStopActions()
            if (isActivityResumed) applyOperatingModeConfiguration()
        }
    }

    private fun runAfterStopActions() {
        if (afterStopActions.isEmpty()) return
        val actions = afterStopActions.toList()
        afterStopActions.clear()
        actions.forEach { action -> action() }
    }

    // Process the current preview frame through VideoProcessor
    private fun processFrameWithVideoProcessor() {
        // Prevent overlapping frame processing
        if (isProcessingFrame) return

        val bitmap = captureProcessingFrame() ?: return
        isProcessingFrame = true

        // Process current frame and update preview output
        videoProcessor?.processFrame(bitmap) { processedBitmap ->
            runOnUiThread {
                if (processedBitmap != null && isProcessing) {
                    displayedProcessedBitmap?.takeIf { it !== processedBitmap }?.recycle()
                    displayedProcessedBitmap = processedBitmap
                    viewBinding.processedFrameView.setImageBitmap(processedBitmap)
                } else {
                    processedBitmap?.recycle()
                }
                isProcessingFrame = false // unlock next frame
            }
        } ?: run {
            bitmap.recycle()
            isProcessingFrame = false // reset if processor is null
        }
    }

    /**
     * TextureView.bitmap uses the full window dimensions, even when the camera buffer is much
     * smaller. Capping the processing copy prevents high-resolution and large-screen devices from
     * allocating multiple 15-30 MB bitmaps for every frame; the models receive no extra detail
     * beyond the camera's 720p preview and YOLO's 640px input.
     */
    private fun captureProcessingFrame(): Bitmap? {
        val preview = viewBinding.viewFinder
        val width = preview.width
        val height = preview.height
        if (width <= 0 || height <= 0) return null

        val scale = (MAX_PROCESSING_FRAME_EDGE.toFloat() / maxOf(width, height))
            .coerceAtMost(1f)
        val targetWidth = (width * scale).roundToInt().coerceAtLeast(1)
        val targetHeight = (height * scale).roundToInt().coerceAtLeast(1)
        return preview.getBitmap(targetWidth, targetHeight)
    }

    // Toggle between front and back camera
    private fun switchCamera() {
        if (isRecording || isStopping) {
            stopProcessingAndRecording(::switchCameraNow)
            return
        }
        switchCameraNow()
    }

    @SuppressLint("MissingPermission")
    private fun switchCameraNow() {
        if (!allPermissionsGranted()) {
            requestRequiredPermissions()
            return
        }
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        if (isActivityResumed) cameraHelper.openCamera()
    }

    // Take actions when you get back to the app
    @SuppressLint("MissingPermission")
    override fun onResume() {
        super.onResume()
        isActivityResumed = true
        cameraHelper.startBackgroundThread()
        applyOperatingModeConfiguration()

        if (viewBinding.viewFinder.isAvailable) {
            if (allPermissionsGranted()) cameraHelper.openCamera() else requestRequiredPermissions()
        }

    }

    // Take actions when another app was opened
    override fun onPause() {
        isActivityResumed = false
        cancelStartCountdown()
        if (arModeManager.isArMode) {
            arModeManager.exitArMode()
        }
        if (isRecording) {
            stopProcessingAndRecording()
        }

        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()

        super.onPause()
    }

    override fun onDestroy() {
        afterStopActions.clear()
        startCountdown?.cancel()
        startCountdown = null
        isCountingDown = false
        collectionDialog?.dismiss()
        collectionDialog = null
        viewBinding.processedFrameView.setImageBitmap(null)
        displayedProcessedBitmap?.recycle()
        displayedProcessedBitmap = null
        emotionInference.close()
        videoProcessor?.close()
        videoProcessor = null
        cameraHelper.release()
        super.onDestroy()
    }

    // Check whether all required permissions are granted
    private fun allPermissionsGranted(): Boolean {
        return requiredPermissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestRequiredPermissions() {
        if (isPermissionRequestInFlight || allPermissionsGranted()) return
        isPermissionRequestInFlight = true
        requestPermissionLauncher.launch(requiredPermissions)
    }

}
