package com.developer27.xemotion.ar

import android.app.Activity
import android.content.res.ColorStateList
import android.os.CountDownTimer
import android.text.InputType
import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.developer27.xemotion.R
import com.developer27.xemotion.camera.CameraHelper
import com.developer27.xemotion.databinding.ActivityMainBinding

/** Controls the immersive AR presentation used by the inference pipeline. */
class ArModeManager(
    private val activity: Activity,
    private val viewBinding: ActivityMainBinding,
    private val cameraHelper: CameraHelper,
    private val isProcessing: () -> Boolean,
    private val onStartProcessing: () -> Unit,
    private val onStopProcessing: () -> Unit
) {
    var isArMode = false
        private set

    private var arTimer: CountDownTimer? = null
    private var startedProcessingForAr = false

    fun promptAndEnterArMode() {
        val margin = (16 * activity.resources.displayMetrics.density).toInt()
        val input = EditText(activity).apply {
            hint = "Minutes"
            inputType = InputType.TYPE_CLASS_NUMBER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(margin, margin, margin, 0) }
        }

        AlertDialog.Builder(activity)
            .setTitle("AR Session Duration")
            .setMessage("Enter how many minutes to run Inference Mode in AR:")
            .setView(input)
            .setPositiveButton("Start") { _, _ ->
                val minutes = input.text.toString().toLongOrNull()
                    ?.coerceIn(1L, MAX_AR_MINUTES)
                    ?: 1L
                enterArMode(minutes * MILLIS_PER_MINUTE)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    fun enterArMode(durationMillis: Long) {
        if (isArMode) return
        isArMode = true
        startedProcessingForAr = !isProcessing()
        if (startedProcessingForAr) onStartProcessing()

        updateArButtonUi(true)
        setArUiVisibility(true)
        hideSystemBars()
        cameraHelper.forceArRollingShutter()

        arTimer?.cancel()
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

        if (startedProcessingForAr && isProcessing()) onStopProcessing()
        startedProcessingForAr = false
        cameraHelper.updateShutterSpeed()
        showSystemBars()
        setArUiVisibility(false)
        updateArButtonUi(false)
    }

    private fun hideSystemBars() {
        WindowCompat.getInsetsController(activity.window, activity.window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    private fun showSystemBars() {
        WindowCompat.getInsetsController(activity.window, activity.window.decorView)
            .show(WindowInsetsCompat.Type.systemBars())
    }

    private fun setArUiVisibility(enabled: Boolean) {
        viewBinding.titleContainer.visibility = if (enabled) View.GONE else View.VISIBLE
        viewBinding.settingsButton.visibility = if (enabled) View.GONE else View.VISIBLE
        viewBinding.aboutButton.visibility = if (enabled) View.GONE else View.VISIBLE
        viewBinding.zoomButtonContainer.visibility = if (enabled) View.GONE else View.VISIBLE
        viewBinding.buttonContainer.visibility = if (enabled) View.GONE else View.VISIBLE
        viewBinding.processedFrameView.visibility = if (enabled || isProcessing()) View.VISIBLE else View.GONE
    }

    private fun updateArButtonUi(enabled: Boolean) {
        viewBinding.arModeButton.apply {
            text = if (enabled) "Exit AR" else "AR Mode"
            backgroundTintList = ColorStateList.valueOf(
                ContextCompat.getColor(activity, if (enabled) R.color.red else R.color.green)
            )
            setTextColor(ContextCompat.getColor(activity, android.R.color.white))
        }
    }

    private companion object {
        const val MILLIS_PER_MINUTE = 60_000L
        const val MAX_AR_MINUTES = 24L * 60L
    }
}
