package com.developer27.xemotion.ar

import android.content.pm.ActivityInfo
import android.content.res.Configuration
import android.os.CountDownTimer
import android.text.InputType
import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.preference.PreferenceManager
import com.developer27.xemotion.R
import com.developer27.xemotion.camera.CameraHelper
import com.developer27.xemotion.databinding.ActivityMainBinding

/** Owns one foreground-only headset session and restores the handheld presentation on exit. */
class ArModeManager(
    private val activity: AppCompatActivity,
    private val viewBinding: ActivityMainBinding,
    private val cameraHelper: CameraHelper,
    private val isProcessing: () -> Boolean,
    private val onStartProcessing: () -> Unit,
    private val onStopProcessing: () -> Unit
) {
    var isArMode = false
        private set
    private var arTimer: CountDownTimer? = null
    private var setupDialog: AlertDialog? = null
    private var startedProcessingForAr = false
    private var previousOrientation = ActivityInfo.SCREEN_ORIENTATION_FULL_SENSOR
    private var countdown = ""
    private val preferences = PreferenceManager.getDefaultSharedPreferences(activity)
    private val backCallback = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() = exitArMode()
    }

    init {
        activity.onBackPressedDispatcher.addCallback(activity, backCallback)
        viewBinding.cardboardView.setOnLongClickListener { exitArMode(); true }
    }

    fun promptAndEnterArMode() {
        if (isArMode || setupDialog?.isShowing == true) return
        val margin = (24 * activity.resources.displayMetrics.density).toInt()
        val input = EditText(activity).apply {
            hint = activity.getString(R.string.cardboard_minutes)
            inputType = InputType.TYPE_CLASS_NUMBER
            setText(preferences.getInt("cardboard_minutes", 5).toString())
            isSingleLine = true
        }
        val content = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(margin, 0, margin, 0)
            addView(input)
        }
        val dialog = AlertDialog.Builder(activity)
            .setTitle(R.string.cardboard_title)
            .setMessage(R.string.cardboard_setup)
            .setView(content)
            .setPositiveButton("Start", null)
            .setNegativeButton(android.R.string.cancel, null)
            .create()
        setupDialog = dialog
        dialog.setOnDismissListener { if (setupDialog === dialog) setupDialog = null }
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val minutes = input.text.toString().toLongOrNull()
                if (minutes == null || minutes !in 1L..MAX_AR_MINUTES) {
                    input.error = activity.getString(R.string.cardboard_invalid_duration)
                    return@setOnClickListener
                }
                preferences.edit().putInt("cardboard_minutes", minutes.toInt()).apply()
                dialog.dismiss()
                enterArMode(minutes * MILLIS_PER_MINUTE)
            }
        }
        dialog.show()
    }

    fun enterArMode(durationMillis: Long) {
        if (isArMode || durationMillis <= 0) return
        startedProcessingForAr = !isProcessing()
        // Starting can fail (permissions, camera, OpenCV, model loading, stop in progress).
        // Do not strand the user in a blank immersive session when it does.
        if (startedProcessingForAr) onStartProcessing()
        if (!isProcessing()) {
            startedProcessingForAr = false
            return
        }
        isArMode = true
        previousOrientation = activity.requestedOrientation
        activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        backCallback.isEnabled = true
        cameraHelper.refreshCameraSettings()
        viewBinding.cardboardView.setFrame(null)
        countdown = viewBinding.countdownText.text.toString()
        refreshUi()
        arTimer = object : CountDownTimer(durationMillis, 1_000L) {
            override fun onTick(millisUntilFinished: Long) = Unit
            override fun onFinish() = exitArMode()
        }.start()
    }

    fun exitArMode() {
        if (!isArMode) return
        isArMode = false
        arTimer?.cancel()
        arTimer = null
        backCallback.isEnabled = false
        countdown = ""
        viewBinding.cardboardView.setFrame(null)
        if (startedProcessingForAr && isProcessing()) onStopProcessing()
        startedProcessingForAr = false
        cameraHelper.refreshCameraSettings()
        activity.requestedOrientation = previousOrientation
        refreshUi()
        viewBinding.processedFrameView.visibility = if (isProcessing()) View.VISIBLE else View.GONE
        WindowCompat.getInsetsController(activity.window, activity.window.decorView)
            .show(WindowInsetsCompat.Type.systemBars())
    }

    fun setCountdown(text: String) {
        countdown = text
        if (isArMode) updateStatus()
    }

    /** Call after ordinary UI updates and after size/configuration changes. */
    fun refreshUi() {
        viewBinding.titleContainer.visibility = if (isArMode) View.GONE else View.VISIBLE
        viewBinding.topControlsContainer.visibility = if (isArMode) View.GONE else View.VISIBLE
        viewBinding.zoomButtonContainer.visibility = if (isArMode) View.GONE else View.VISIBLE
        viewBinding.buttonContainer.visibility = if (isArMode) View.GONE else View.VISIBLE
        viewBinding.cardboardView.visibility = if (isArMode) View.VISIBLE else View.GONE
        if (isArMode) {
            viewBinding.processedFrameView.visibility = View.GONE
            viewBinding.countdownText.visibility = View.GONE
            hideSystemBars()
            updateStatus()
        }
    }

    fun hideSystemBars() {
        if (!isArMode) return
        WindowCompat.getInsetsController(activity.window, activity.window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    private fun updateStatus() {
        viewBinding.cardboardView.statusText = when {
            activity.resources.configuration.orientation != Configuration.ORIENTATION_LANDSCAPE ->
                activity.getString(R.string.cardboard_rotate)
            countdown.isNotEmpty() -> activity.getString(R.string.cardboard_starting, countdown)
            else -> ""
        }
    }

    fun onPause() {
        setupDialog?.dismiss()
        setupDialog = null
        exitArMode()
    }

    private companion object {
        const val MILLIS_PER_MINUTE = 60_000L
        const val MAX_AR_MINUTES = 24L * 60L
    }
}
