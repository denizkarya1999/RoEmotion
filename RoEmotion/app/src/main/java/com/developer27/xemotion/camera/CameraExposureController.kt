package com.developer27.xemotion.camera

import android.content.SharedPreferences
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import kotlin.math.max

/** Applies exposure, lighting, and flash preferences to a capture request. */
class CameraExposureController(
    private val cameraManager: CameraManager,
    private val preferences: SharedPreferences,
    private val cameraIdProvider: () -> String
) {
    fun applyRollingShutter(builder: CaptureRequest.Builder?) {
        builder ?: return
        val characteristics = cameraManager.getCameraCharacteristics(cameraIdProvider())
        if (!characteristics.supportsManualExposure()) {
            setAutoExposure(builder)
            return
        }

        val shutterFrequency = preferences.getString(
            KEY_SHUTTER_SPEED,
            DEFAULT_SHUTTER_FREQUENCY.toString()
        )
            ?.toIntOrNull()
            ?: DEFAULT_SHUTTER_FREQUENCY
        val requestedExposure = shutterFrequency
            .takeIf { it > 0 }
            ?.let { NANOS_PER_SECOND / it }

        val exposureRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
        if (requestedExposure == null || exposureRange == null || isoRange == null) {
            setAutoExposure(builder)
            return
        }

        val requestedIso = preferences.getString(KEY_ISO_VALUE, DEFAULT_ISO.toString())
            ?.toIntOrNull()
            ?: DEFAULT_ISO
        val iso = if (preferences.getBoolean(KEY_MANUAL_ISO, true)) {
            requestedIso.coerceIn(isoRange.lower, isoRange.upper)
        } else {
            ((isoRange.lower + isoRange.upper) / 2).coerceIn(isoRange.lower, isoRange.upper)
        }

        setManualExposure(
            builder,
            requestedExposure.coerceIn(exposureRange.lower, exposureRange.upper),
            iso
        )
    }

    fun applyArExposure(builder: CaptureRequest.Builder?): Boolean {
        builder ?: return false
        val characteristics = cameraManager.getCameraCharacteristics(cameraIdProvider())
        if (!characteristics.supportsManualExposure()) return false

        val exposureRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            ?: return false
        val isoRange = characteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            ?: return false
        val exposure = (NANOS_PER_SECOND / AR_SHUTTER_FREQUENCY)
            .coerceIn(exposureRange.lower, exposureRange.upper)
        val iso = max(isoRange.lower, AR_ISO).coerceAtMost(isoRange.upper)
        setManualExposure(builder, exposure, iso)
        return true
    }

    fun applyFlash(builder: CaptureRequest.Builder?) {
        builder?.set(
            CaptureRequest.FLASH_MODE,
            if (preferences.getBoolean(KEY_FLASH, false)) {
                CaptureRequest.FLASH_MODE_TORCH
            } else {
                CaptureRequest.FLASH_MODE_OFF
            }
        )
    }

    fun applyLighting(builder: CaptureRequest.Builder?) {
        builder ?: return
        if (builder.get(CaptureRequest.CONTROL_AE_MODE) != CameraMetadata.CONTROL_AE_MODE_ON) return

        val range = cameraManager
            .getCameraCharacteristics(cameraIdProvider())
            .get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
        val compensation = when (preferences.getString(KEY_LIGHTING_MODE, NORMAL_LIGHTING)) {
            LOW_LIGHTING -> range?.lower ?: 0
            HIGH_LIGHTING -> range?.upper ?: 0
            else -> 0
        }
        builder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, compensation)
    }

    private fun CameraCharacteristics.supportsManualExposure(): Boolean =
        get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.contains(
            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR
        ) == true

    private fun setAutoExposure(builder: CaptureRequest.Builder) {
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
    }

    private fun setManualExposure(builder: CaptureRequest.Builder, exposureNanos: Long, iso: Int) {
        builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
        builder.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposureNanos)
        builder.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
    }

    private companion object {
        const val NANOS_PER_SECOND = 1_000_000_000L
        const val DEFAULT_SHUTTER_FREQUENCY = 6000
        const val DEFAULT_ISO = 400
        const val AR_SHUTTER_FREQUENCY = 6000
        const val AR_ISO = 100
        const val KEY_SHUTTER_SPEED = "shutter_speed"
        const val KEY_MANUAL_ISO = "manual_iso_enabled"
        const val KEY_ISO_VALUE = "iso_value"
        const val KEY_FLASH = "enable_flash"
        const val KEY_LIGHTING_MODE = "lighting_mode"
        const val NORMAL_LIGHTING = "normal"
        const val LOW_LIGHTING = "low_light"
        const val HIGH_LIGHTING = "high_light"
    }
}
