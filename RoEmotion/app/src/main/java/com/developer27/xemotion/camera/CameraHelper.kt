package com.developer27.xemotion.camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.MotionEvent
import android.view.Surface
import android.widget.Toast
import androidx.annotation.RequiresPermission
import com.developer27.xemotion.MainActivity
import com.developer27.xemotion.databinding.ActivityMainBinding
import kotlin.math.max

/**
 * CameraHelper is responsible for:
 *  - Opening & closing the camera
 *  - Switching front/back
 *  - Creating a preview
 *  - Handling zoom & shutter speed
 *  - Starting a background thread for camera operations
 *
 *  This version forces a specific AWB mode & color correction to avoid color tint on Pixel 4a.
 */
class CameraHelper(
    private val activity: MainActivity,
    private val viewBinding: ActivityMainBinding,
    private val sharedPreferences: SharedPreferences
) {
    // The Android Camera2 API
    val cameraManager: CameraManager by lazy {
        activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    // Active camera device + capture session
    var cameraDevice: CameraDevice? = null
    var cameraCaptureSession: CameraCaptureSession? = null

    // Capture builder for preview (and record)
    var captureRequestBuilder: CaptureRequest.Builder? = null

    // Preview + video sizes
    var previewSize: Size? = null
    var videoSize: Size? = null

    // Sensor area for zoom
    var sensorArraySize: Rect? = null

    // Whether we are using the front camera
    var isFrontCamera = false

    // Thread for camera operations
    private var backgroundThread: HandlerThread? = null
    var backgroundHandler: Handler? = null
        private set

    // Zoom control
    private var zoomLevel = 1.0f
    private val maxZoom = 10.0f

    /**
     * Callback for camera device events
     */
    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            // When camera is opened, store reference and create preview
            cameraDevice = camera
            createCameraPreview()
        }

        override fun onDisconnected(camera: CameraDevice) {
            // Close camera if disconnected
            cameraDevice?.close()
            cameraDevice = null
        }

        override fun onError(camera: CameraDevice, error: Int) {
            // Close on errors
            cameraDevice?.close()
            cameraDevice = null
            activity.finish()
        }
    }

    // ------------------------------------------------------------------------
    // Background Thread Setup
    // ------------------------------------------------------------------------
    fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Open/Close Camera
    // ------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    @RequiresPermission(Manifest.permission.CAMERA)
    fun openCamera() {
        try {
            // Decide which camera (front/back)
            val cameraId = getCameraId()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // Grab the full sensor area for zoom
            sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)

            // Possible output sizes
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: return

            // Choose your preview/video sizes
            previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
            videoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))

            // Now open the selected camera
            cameraManager.openCamera(cameraId, stateCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: SecurityException) {
            e.printStackTrace()
            Toast.makeText(activity, "Camera permission needed.", Toast.LENGTH_SHORT).show()
        }
    }

    fun closeCamera() {
        cameraCaptureSession?.close()
        cameraCaptureSession = null
        cameraDevice?.close()
        cameraDevice = null
    }

    // ------------------------------------------------------------------------
    // Create Preview
    // ------------------------------------------------------------------------
    fun createCameraPreview() {
        try {
            val texture = viewBinding.viewFinder.surfaceTexture ?: return
            // Match the texture view size to the chosen preview size
            previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }

            val previewSurface = Surface(texture)
            // Build a preview request
            captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            // Add the preview surface as a target
            captureRequestBuilder?.addTarget(previewSurface)

            // Apply any manual or auto exposure logic
            applyRollingShutter()
            // Possibly set flash, lighting, zoom
            applyFlashIfEnabled()
            applyLightingMode()
            applyZoom()

            // ----------------------------------------------------------------
            // Force color correction to avoid greenish tint
            // 1) Auto White Balance (set to e.g. DAYLIGHT for consistent color)
            //    or CONTROL_AWB_MODE_AUTO for auto
            // 2) Color Correction Mode => HIGH_QUALITY for better color
            // ----------------------------------------------------------------
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AWB_MODE,
                // For strictly "daylight" color:
                // CaptureRequest.CONTROL_AWB_MODE_DAYLIGHT
                // or if you prefer auto, do:
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            )
            captureRequestBuilder?.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
            )

            // Now create the capture session
            cameraDevice?.createCaptureSession(
                listOf(previewSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice == null) return
                        // Save the session
                        cameraCaptureSession = session
                        updatePreview() // Start the preview
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(
                            activity,
                            "Preview config failed.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    /**
     * Update the camera preview with latest builder settings
     */
    fun updatePreview() {
        if (cameraDevice == null || captureRequestBuilder == null) return
        try {
            // Keep forcing color correction and AWB
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AWB_MODE,
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            )
            captureRequestBuilder?.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
            )

            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Camera Selection (Front/Back)
    // ------------------------------------------------------------------------
    fun getCameraId(): String {
        for (id in cameraManager.cameraIdList) {
            val facing = cameraManager
                .getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (!isFrontCamera && facing == CameraCharacteristics.LENS_FACING_BACK) {
                return id
            } else if (isFrontCamera && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                return id
            }
        }
        // fallback if none matched
        return cameraManager.cameraIdList.first()
    }

    private fun chooseOptimalSize(choices: Array<Size>): Size {
        val targetWidth = 1280
        val targetHeight = 720

        // Try to find 1280x720 specifically
        val found720p = choices.find { it.width == targetWidth && it.height == targetHeight }
        if (found720p != null) {
            return found720p
        }
        // fallback to the smallest
        return choices.minByOrNull { it.width * it.height } ?: choices[0]
    }

    // ------------------------------------------------------------------------
    // Rolling shutter & exposure
    // ------------------------------------------------------------------------
    fun applyRollingShutter() {
        val cameraId = getCameraId()
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)

        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val canManualExposure = capabilities?.contains(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR
        ) == true

        val shutterFps = sharedPreferences.getString("shutter_speed", "15")?.toIntOrNull() ?: 15
        val shutterValueNs = if (shutterFps > 0) 1_000_000_000L / shutterFps else 0L

        // If no manual or user set 0, just do auto
        if (!canManualExposure || shutterValueNs <= 0) {
            setAutoExposure()
            return
        }

        val exposureTimeRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)

        if (exposureTimeRange == null || isoRange == null) {
            setAutoExposure()
            return
        }

        val safeExposureNs = shutterValueNs.coerceIn(exposureTimeRange.lower, exposureTimeRange.upper)

        // Read ISO prefs
        val manualIsoEnabled = sharedPreferences.getBoolean("manual_iso_enabled", true)
        val isoFromPrefs = sharedPreferences.getString("iso_value", "800")?.toIntOrNull() ?: 800
        val safeISO = isoFromPrefs.coerceIn(isoRange.lower, isoRange.upper)

        // Fully manual exposure; if manual ISO disabled, we still must set *some* ISO because AE is off.
        // Choose either the user ISO or a mid-range fallback.
        val isoToUse = if (manualIsoEnabled) safeISO else
            ((isoRange.lower + isoRange.upper) / 2).coerceIn(isoRange.lower, isoRange.upper)

        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_EXPOSURE_TIME, safeExposureNs)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_SENSITIVITY, isoToUse)
    }

    private fun setAutoExposure() {
        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
    }

    fun forceArRollingShutter() {
        // For AR mode, fix shutter speed to 350 Hz (1 / 350s).
        // Then immediately update the camera capture session.

        val cameraId = getCameraId()
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        val capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
        val canManualExposure = capabilities?.contains(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR
        ) == true

        // If we can’t do manual exposure or the camera is null, just return
        if (!canManualExposure || captureRequestBuilder == null) return

        val exposureTimeRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        if (exposureTimeRange == null || isoRange == null) return

        // Convert 250 Hz
        val forcedShutterNs = 1_000_000_000L / 6000
        val safeExposureNs = forcedShutterNs.coerceIn(exposureTimeRange.lower, exposureTimeRange.upper)
        val safeISO = max(isoRange.lower, 100)

        // Apply fully manual
        captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_EXPOSURE_TIME, safeExposureNs)
        captureRequestBuilder?.set(CaptureRequest.SENSOR_SENSITIVITY, safeISO)

        // Now push the settings to the active capture session
        try {
            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    /**
     * If user changes shutter speed in settings, we re-apply
     */
    fun updateShutterSpeed() {
        applyRollingShutter()
        try {
            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------
    // Flash & Lighting
    // ------------------------------------------------------------------------
    fun applyFlashIfEnabled() {
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)
        captureRequestBuilder?.set(
            CaptureRequest.FLASH_MODE,
            if (isFlashEnabled) CaptureRequest.FLASH_MODE_TORCH
            else CaptureRequest.FLASH_MODE_OFF
        )
    }

    fun applyLightingMode() {
        // Only apply AE compensation if AE is ON
        val aeMode = captureRequestBuilder?.get(CaptureRequest.CONTROL_AE_MODE)
        if (aeMode == CameraMetadata.CONTROL_AE_MODE_ON) {
            val lightingMode = sharedPreferences.getString("lighting_mode", "normal")
            val cameraId = getCameraId()
            val compensationRange = cameraManager
                .getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)

            val exposureComp = when (lightingMode) {
                "low_light" -> compensationRange?.lower ?: 0
                "high_light" -> compensationRange?.upper ?: 0
                else -> 0
            }
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
                exposureComp
            )
        }
    }

    // ------------------------------------------------------------------------
    // Zoom
    // ------------------------------------------------------------------------
    fun setupZoomControls() {
        val zoomHandler = Handler(activity.mainLooper)
        var zoomInRunnable: Runnable? = null
        var zoomOutRunnable: Runnable? = null

        // Repetitive zoom in on long-press
        viewBinding.zoomInButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomInRunnable = object : Runnable {
                        override fun run() {
                            zoomIn()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomInRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomHandler.removeCallbacks(zoomInRunnable!!)
                    true
                }
                else -> false
            }
        }

        // Repetitive zoom out on long-press
        viewBinding.zoomOutButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomOutRunnable = object : Runnable {
                        override fun run() {
                            zoomOut()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomOutRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomHandler.removeCallbacks(zoomOutRunnable!!)
                    true
                }
                else -> false
            }
        }
    }

    private fun zoomIn() {
        if (zoomLevel < maxZoom) {
            zoomLevel += 0.1f
            applyZoom()
        }
    }

    private fun zoomOut() {
        if (zoomLevel > 1.0f) {
            zoomLevel -= 0.1f
            applyZoom()
        }
    }

    /**
     * Applies digital zoom by setting the SCALER_CROP_REGION
     */
    fun applyZoom() {
        if (sensorArraySize == null || captureRequestBuilder == null) return
        val ratio = 1 / zoomLevel
        val croppedWidth = sensorArraySize!!.width() * ratio
        val croppedHeight = sensorArraySize!!.height() * ratio

        val left = ((sensorArraySize!!.width() - croppedWidth) / 2).toInt()
        val top = ((sensorArraySize!!.height() - croppedHeight) / 2).toInt()
        val right = (left + croppedWidth).toInt()
        val bottom = (top + croppedHeight).toInt()

        val zoomRect = Rect(left, top, right, bottom)
        captureRequestBuilder?.set(CaptureRequest.SCALER_CROP_REGION, zoomRect)

        try {
            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }
}