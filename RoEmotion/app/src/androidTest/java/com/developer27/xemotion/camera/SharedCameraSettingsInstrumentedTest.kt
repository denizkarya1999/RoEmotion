package com.developer27.xemotion.camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CaptureRequest
import android.os.Handler
import android.os.HandlerThread
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.developer27.xemotion.ar.CardboardTestActivity
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class SharedCameraSettingsInstrumentedTest {
    @SuppressLint("MissingPermission")
    @Test fun savedCameraSettingsSurviveArEntryChangesAndExit() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val preferences = instrumentation.targetContext
            .getSharedPreferences("cardboard_test_camera", Context.MODE_PRIVATE)
        val thread = HandlerThread("CameraSettingsTest").apply { start() }
        var camera: CameraDevice? = null
        instrumentation.uiAutomation.adoptShellPermissionIdentity(Manifest.permission.CAMERA)
        try {
            ActivityScenario.launch(CardboardTestActivity::class.java).use { scenario ->
                val opened = CompletableFuture<CameraDevice>()
                lateinit var characteristics: CameraCharacteristics
                scenario.onActivity { activity ->
                    val helper = activity.cameraHelper
                    val id = helper.getCameraId()
                    characteristics = helper.cameraManager.getCameraCharacteristics(id)
                    helper.cameraManager.openCamera(id, object : CameraDevice.StateCallback() {
                        override fun onOpened(device: CameraDevice) { opened.complete(device) }
                        override fun onDisconnected(device: CameraDevice) {
                            device.close()
                            opened.completeExceptionally(IllegalStateException("Camera disconnected"))
                        }
                        override fun onError(device: CameraDevice, error: Int) {
                            device.close()
                            opened.completeExceptionally(IllegalStateException("Camera error $error"))
                        }
                    }, Handler(thread.looper))
                }
                val device = opened.get(10, TimeUnit.SECONDS)
                camera = device
                val isoRange = requireNotNull(characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE))
                val exposureRange = requireNotNull(characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE))
                scenario.onActivity { activity ->
                    val helper = activity.cameraHelper
                    fun assertManual(frequency: Int, iso: Int, flash: Boolean) {
                        val request = requireNotNull(helper.captureRequestBuilder)
                        assertEquals(CaptureRequest.CONTROL_AE_MODE_OFF, request.get(CaptureRequest.CONTROL_AE_MODE))
                        assertEquals(exposureRange.clamp(1_000_000_000L / frequency), request.get(CaptureRequest.SENSOR_EXPOSURE_TIME))
                        assertEquals(isoRange.clamp(iso), request.get(CaptureRequest.SENSOR_SENSITIVITY))
                        assertEquals(if (flash) CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF,
                            request.get(CaptureRequest.FLASH_MODE))
                    }
                    preferences.edit().clear().putString("shutter_speed", "3000")
                        .putString("iso_value", "800").putBoolean("manual_iso_enabled", true).commit()
                    helper.captureRequestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    helper.refreshCameraSettings()
                    assertManual(3000, 800, false)
                    activity.manager.enterArMode(60000)
                    assertTrue(activity.manager.isArMode)
                    assertManual(3000, 800, false)

                    preferences.edit().putString("shutter_speed", "12000").putString("iso_value", "1600")
                        .putBoolean("enable_flash", true).commit()
                    helper.refreshCameraSettings()
                    assertManual(12000, 1600, true)
                    // A replacement request while in AR also uses saved values.
                    helper.captureRequestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    helper.refreshCameraSettings()
                    assertManual(12000, 1600, true)
                    activity.manager.exitArMode()
                    assertManual(12000, 1600, true)

                    preferences.edit().putBoolean("manual_iso_enabled", false).commit()
                    activity.manager.enterArMode(60000)
                    assertManual(12000, (isoRange.lower + isoRange.upper) / 2, true)
                    preferences.edit().putString("shutter_speed", "0")
                        .putString("lighting_mode", "high_light").putBoolean("enable_flash", false).commit()
                    helper.refreshCameraSettings()
                    assertEquals(CaptureRequest.CONTROL_AE_MODE_ON,
                        helper.captureRequestBuilder?.get(CaptureRequest.CONTROL_AE_MODE))
                    assertEquals(characteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)?.upper ?: 0,
                        helper.captureRequestBuilder?.get(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION))
                    assertEquals(CaptureRequest.FLASH_MODE_OFF, helper.captureRequestBuilder?.get(CaptureRequest.FLASH_MODE))
                    activity.manager.exitArMode()
                    helper.captureRequestBuilder = null
                }
            }
        } finally {
            camera?.close()
            thread.quitSafely()
            thread.join(3000)
            instrumentation.uiAutomation.dropShellPermissionIdentity()
            preferences.edit().clear().commit()
        }
    }
}
