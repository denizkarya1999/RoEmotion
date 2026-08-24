package com.developer27.xemotion.camera

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface
import android.widget.Toast
import androidx.annotation.RequiresPermission
import com.developer27.xemotion.databinding.ActivityMainBinding
import java.util.concurrent.Executor

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
    private val activity: Activity,
    private val viewBinding: ActivityMainBinding,
    private val sharedPreferences: SharedPreferences
) {
    // The Android Camera2 API
    val cameraManager: CameraManager by lazy {
        activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    private val exposureController by lazy {
        CameraExposureController(cameraManager, sharedPreferences, ::getCameraId)
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

    private var zoomController: ZoomController? = null
    private var currentZoom = 1f
    private val cameraStateLock = Any()
    private var cameraRequested = false
    private var requestedCameraId: String? = null
    private var openingCameraId: String? = null
    private var previewSurface: Surface? = null
    private var previewSessionPending = false

    /**
     * Callback for camera device events
     */
    private val stateCallback = object : CameraDevice.StateCallback() {
        @SuppressLint("MissingPermission")
        override fun onOpened(camera: CameraDevice) {
            val useCamera = synchronized(cameraStateLock) {
                openingCameraId = null
                if (cameraRequested && requestedCameraId == camera.id) {
                    cameraDevice = camera
                    true
                } else {
                    false
                }
            }
            if (useCamera) {
                createCameraPreview()
            } else {
                camera.close()
                if (synchronized(cameraStateLock) { cameraRequested }) openCamera()
            }
        }

        @SuppressLint("MissingPermission")
        override fun onDisconnected(camera: CameraDevice) {
            camera.close()
            val session: CameraCaptureSession?
            val surface: Surface?
            val shouldRetry = synchronized(cameraStateLock) {
                if (openingCameraId == camera.id) openingCameraId = null
                if (cameraDevice === camera) {
                    cameraDevice = null
                    session = cameraCaptureSession
                    cameraCaptureSession = null
                    surface = previewSurface
                    previewSurface = null
                    captureRequestBuilder = null
                    previewSessionPending = false
                } else {
                    session = null
                    surface = null
                }
                cameraRequested
            }
            session?.close()
            surface?.release()
            if (shouldRetry) openCamera()
        }

        @SuppressLint("MissingPermission")
        override fun onError(camera: CameraDevice, error: Int) {
            // Close on errors
            Log.e(TAG, "Camera device error: $error")
            camera.close()
            val session: CameraCaptureSession?
            val surface: Surface?
            val shouldRetry = synchronized(cameraStateLock) {
                if (openingCameraId == camera.id) openingCameraId = null
                val isRequestedCamera = requestedCameraId == camera.id
                if (isRequestedCamera) cameraRequested = false
                if (cameraDevice === camera) {
                    cameraDevice = null
                    session = cameraCaptureSession
                    cameraCaptureSession = null
                    surface = previewSurface
                    previewSurface = null
                    captureRequestBuilder = null
                    previewSessionPending = false
                } else {
                    session = null
                    surface = null
                }
                cameraRequested
            }
            session?.close()
            surface?.release()
            activity.runOnUiThread {
                Toast.makeText(activity, "Camera error ($error). Please try again.", Toast.LENGTH_SHORT).show()
            }
            if (shouldRetry) openCamera()
        }
    }

    // ------------------------------------------------------------------------
    // Background Thread Setup
    // ------------------------------------------------------------------------
    fun startBackgroundThread() {
        if (backgroundThread?.isAlive == true) return

        HandlerThread("CameraBackground").also { thread ->
            thread.start()
            backgroundThread = thread
            backgroundHandler = Handler(thread.looper)
        }
    }

    fun stopBackgroundThread() {
        val thread = backgroundThread ?: return
        thread.quitSafely()
        try {
            thread.join()
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            Log.w(TAG, "Interrupted while stopping the camera thread", e)
        } finally {
            if (backgroundThread === thread) {
                backgroundThread = null
                backgroundHandler = null
            }
        }
    }

    // ------------------------------------------------------------------------
    // Open/Close Camera
    // ------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    @RequiresPermission(Manifest.permission.CAMERA)
    fun openCamera() {
        val cameraId = try {
            getCameraId()
        } catch (error: CameraAccessException) {
            Log.e(TAG, "Unable to select a camera", error)
            return
        } catch (error: IllegalStateException) {
            Log.e(TAG, "No usable camera is available", error)
            showToast(error.message ?: "No usable camera is available.")
            return
        }
        val action = synchronized(cameraStateLock) {
            cameraRequested = true
            requestedCameraId = cameraId
            when {
                cameraDevice?.id == cameraId -> OPEN_ACTION_CREATE_PREVIEW
                openingCameraId != null -> OPEN_ACTION_NONE
                else -> {
                    openingCameraId = cameraId
                    OPEN_ACTION_OPEN_DEVICE
                }
            }
        }
        if (action == OPEN_ACTION_NONE) return
        if (action == OPEN_ACTION_CREATE_PREVIEW) {
            createCameraPreview()
            return
        }

        var submitted = false
        try {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)

            // Grab the full sensor area for zoom
            sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)

            // Possible output sizes
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: error("Camera $cameraId has no stream configuration map")

            // Choose your preview/video sizes
            previewSize = chooseOptimalSize(
                checkNotNull(map.getOutputSizes(SurfaceTexture::class.java)) {
                    "The camera reported no preview output sizes."
                }
            )
            viewBinding.viewFinder.post {
                configurePreviewTransform(
                    viewBinding.viewFinder.width,
                    viewBinding.viewFinder.height
                )
            }
            videoSize = chooseOptimalSize(
                checkNotNull(map.getOutputSizes(MediaRecorder::class.java)) {
                    "The camera reported no video output sizes."
                }
            )

            // Now open the selected camera
            cameraManager.openCamera(cameraId, stateCallback, backgroundHandler)
            submitted = true
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Unable to open the camera", e)
            showToast("Unable to open the camera.")
        } catch (e: SecurityException) {
            Log.e(TAG, "Camera permission is missing", e)
            showToast("Camera permission needed.")
        } catch (e: IllegalStateException) {
            Log.e(TAG, "Camera configuration is unavailable", e)
            showToast("Camera configuration is unavailable.")
        } finally {
            if (!submitted) {
                synchronized(cameraStateLock) {
                    if (openingCameraId == cameraId) openingCameraId = null
                }
            }
        }
    }

    fun closeCamera() {
        val session: CameraCaptureSession?
        val device: CameraDevice?
        val surface: Surface?
        synchronized(cameraStateLock) {
            cameraRequested = false
            requestedCameraId = null
            session = cameraCaptureSession
            cameraCaptureSession = null
            device = cameraDevice
            cameraDevice = null
            surface = previewSurface
            previewSurface = null
            captureRequestBuilder = null
            previewSessionPending = false
        }
        session?.close()
        device?.close()
        surface?.release()
    }

    // ------------------------------------------------------------------------
    // Create Preview
    // ------------------------------------------------------------------------
    fun createCameraPreview() {
        val texture = viewBinding.viewFinder.surfaceTexture ?: return
        val device = synchronized(cameraStateLock) {
            if (!cameraRequested || cameraCaptureSession != null || previewSessionPending) return
            previewSessionPending = true
            cameraDevice
        } ?: run {
            synchronized(cameraStateLock) { previewSessionPending = false }
            return
        }
        var newSurface: Surface? = null
        var submitted = false
        try {
            // Match the texture view size to the chosen preview size
            previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }

            val surface = Surface(texture)
            newSurface = surface
            // Build a preview request
            val builder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            // Add the preview surface as a target
            builder.addTarget(surface)

            val accepted = synchronized(cameraStateLock) {
                if (!cameraRequested || cameraDevice !== device) {
                    false
                } else {
                    previewSurface?.release()
                    previewSurface = surface
                    captureRequestBuilder = builder
                    true
                }
            }
            if (!accepted) return

            // Apply any manual or auto exposure logic
            applyRollingShutter()
            // Possibly set flash, lighting, zoom
            applyFlashIfEnabled()
            applyLightingMode()
            applyZoom(currentZoom)

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

            // Now create the capture session.
            val sessionCallback = object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    val accepted = synchronized(cameraStateLock) {
                        previewSessionPending = false
                        if (!cameraRequested ||
                            cameraDevice !== session.device ||
                            previewSurface !== surface
                        ) {
                            false
                        } else {
                            cameraCaptureSession?.close()
                            cameraCaptureSession = session
                            true
                        }
                    }
                    if (accepted) {
                        updatePreview()
                    } else {
                        session.close()
                        releasePreviewSurface(surface)
                    }
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    session.close()
                    synchronized(cameraStateLock) { previewSessionPending = false }
                    releasePreviewSurface(surface)
                    Log.e(TAG, "Camera preview configuration failed")
                    showToast("Preview config failed.")
                }
            }
            createCaptureSession(device, surface, sessionCallback)
            submitted = true
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Unable to create the camera preview", e)
        } catch (e: IllegalStateException) {
            Log.e(TAG, "Camera closed while creating preview", e)
        } finally {
            if (!submitted) {
                synchronized(cameraStateLock) { previewSessionPending = false }
                newSurface?.let { surface ->
                    val isCurrent = synchronized(cameraStateLock) { previewSurface === surface }
                    if (isCurrent) releasePreviewSurface(surface) else surface.release()
                }
            }
        }
    }

    private fun createCaptureSession(
        device: CameraDevice,
        surface: Surface,
        callback: CameraCaptureSession.StateCallback
    ) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val callbackExecutor = Executor { command ->
                backgroundHandler?.post(command) ?: activity.runOnUiThread(command)
            }
            device.createCaptureSession(
                SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    listOf(OutputConfiguration(surface)),
                    callbackExecutor,
                    callback
                )
            )
        } else {
            @Suppress("DEPRECATION")
            device.createCaptureSession(listOf(surface), callback, backgroundHandler)
        }
    }

    /**
     * Update the camera preview with latest builder settings
     */
    fun updatePreview() {
        val (session, builder) = synchronized(cameraStateLock) {
            if (cameraDevice == null) return
            (cameraCaptureSession ?: return) to (captureRequestBuilder ?: return)
        }
        try {
            // Keep forcing color correction and AWB
            builder.set(
                CaptureRequest.CONTROL_AWB_MODE,
                CaptureRequest.CONTROL_AWB_MODE_AUTO
            )
            builder.set(
                CaptureRequest.COLOR_CORRECTION_MODE,
                CaptureRequest.COLOR_CORRECTION_MODE_HIGH_QUALITY
            )

            session.setRepeatingRequest(
                builder.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Unable to update the camera preview", e)
        } catch (e: IllegalStateException) {
            Log.w(TAG, "Camera session closed before preview update", e)
        }
    }

    // ------------------------------------------------------------------------
    // Camera Selection (Front/Back)
    // ------------------------------------------------------------------------
    fun getCameraId(): String {
        val cameraIds = cameraManager.cameraIdList
        check(cameraIds.isNotEmpty()) { "No camera is available on this device." }
        for (id in cameraIds) {
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
        return cameraIds.first()
    }

    private fun chooseOptimalSize(choices: Array<Size>): Size {
        check(choices.isNotEmpty()) { "The camera reported no supported output sizes." }
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

    /** Keeps the Camera2 preview correctly cropped and rotated as a window is resized. */
    fun configurePreviewTransform(viewWidth: Int, viewHeight: Int) {
        val size = previewSize ?: return
        if (viewWidth <= 0 || viewHeight <= 0) return

        @Suppress("DEPRECATION")
        val rotation = activity.windowManager.defaultDisplay.rotation
        val matrix = Matrix()
        val centerX = viewWidth / 2f
        val centerY = viewHeight / 2f

        when (rotation) {
            Surface.ROTATION_0 -> Unit

            Surface.ROTATION_90, Surface.ROTATION_270 -> {
                val viewRect = RectF(0f, 0f, viewWidth.toFloat(), viewHeight.toFloat())
                val bufferRect = RectF(0f, 0f, size.height.toFloat(), size.width.toFloat())
                bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())
                matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL)
                val scale = maxOf(
                    viewHeight.toFloat() / size.height,
                    viewWidth.toFloat() / size.width
                )
                matrix.postScale(scale, scale, centerX, centerY)
                matrix.postRotate(90f * (rotation - 2), centerX, centerY)
            }

            Surface.ROTATION_180 -> matrix.postRotate(180f, centerX, centerY)
        }
        viewBinding.viewFinder.setTransform(matrix)
    }

    // ------------------------------------------------------------------------
    // Rolling shutter & exposure
    // ------------------------------------------------------------------------
    fun applyRollingShutter() {
        exposureController.applyRollingShutter(captureRequestBuilder)
    }

    fun forceArRollingShutter() {
        if (exposureController.applyArExposure(captureRequestBuilder)) submitRepeatingRequest()
    }

    /**
     * If user changes shutter speed in settings, we re-apply
     */
    fun updateShutterSpeed() {
        applyRollingShutter()
        submitRepeatingRequest()
    }

    // ------------------------------------------------------------------------
    // Flash & Lighting
    // ------------------------------------------------------------------------
    fun applyFlashIfEnabled() {
        exposureController.applyFlash(captureRequestBuilder)
    }

    fun applyLightingMode() {
        exposureController.applyLighting(captureRequestBuilder)
    }

    // ------------------------------------------------------------------------
    // Zoom
    // ------------------------------------------------------------------------
    fun setupZoomControls() {
        zoomController?.close()
        zoomController = ZoomController(
            zoomInView = viewBinding.zoomInButton,
            zoomOutView = viewBinding.zoomOutButton,
            handler = Handler(activity.mainLooper),
            onZoomChanged = { zoom ->
                currentZoom = zoom
                applyZoom(zoom)
            }
        ).also { it.bind() }
    }

    /**
     * Applies digital zoom by setting the SCALER_CROP_REGION
     */
    private fun applyZoom(zoomLevel: Float) {
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

        submitRepeatingRequest()
    }

    private fun submitRepeatingRequest() {
        val (session, request) = synchronized(cameraStateLock) {
            (cameraCaptureSession ?: return) to (captureRequestBuilder?.build() ?: return)
        }
        try {
            session.setRepeatingRequest(request, null, backgroundHandler)
        } catch (error: CameraAccessException) {
            Log.e(TAG, "Unable to update repeating request", error)
        } catch (error: IllegalStateException) {
            Log.w(TAG, "Camera session closed before request update", error)
        }
    }

    private fun releasePreviewSurface(surface: Surface) {
        val shouldRelease = synchronized(cameraStateLock) {
            if (previewSurface === surface) {
                previewSurface = null
                captureRequestBuilder = null
                true
            } else {
                false
            }
        }
        if (shouldRelease) surface.release()
    }

    private fun showToast(message: String) {
        activity.runOnUiThread {
            Toast.makeText(activity, message, Toast.LENGTH_SHORT).show()
        }
    }

    fun release() {
        zoomController?.close()
        zoomController = null
        closeCamera()
    }

    private companion object {
        const val TAG = "CameraHelper"
        const val OPEN_ACTION_NONE = 0
        const val OPEN_ACTION_CREATE_PREVIEW = 1
        const val OPEN_ACTION_OPEN_DEVICE = 2
    }
}
