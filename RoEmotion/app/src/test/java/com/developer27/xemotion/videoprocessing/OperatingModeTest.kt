package com.developer27.xemotion.videoprocessing

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.text.SimpleDateFormat
import java.util.Locale

class OperatingModeTest {
    @After
    fun restoreDefaultMode() {
        Settings.applyOperatingMode(Settings.OperatingMode.Mode.INFERENCE)
        Settings.Trace.updateCollectionTypes(setOf(Settings.Trace.Type.SPLINE_CV))
        Settings.Trace.updateInferenceType(Settings.Trace.Type.SPLINE_CV)
        Settings.Trace.updateBoldness(Settings.Trace.DEFAULT_BOLDNESS)
        Settings.Inference.updateLabelSize(Settings.Inference.DEFAULT_LABEL_SIZE)
    }

    @Test
    fun dataCollectionModeForcesContourAndAutomaticExport() {
        Settings.applyOperatingMode(Settings.OperatingMode.Mode.DATA_COLLECTION)

        assertEquals(Settings.OperatingMode.Mode.DATA_COLLECTION, Settings.OperatingMode.current)
        assertEquals(Settings.DetectionMode.Mode.CONTOUR, Settings.DetectionMode.current)
        assertFalse(Settings.DetectionMode.enableYOLOinference)
        assertTrue(Settings.ExportData.frameIMG)
    }

    @Test
    fun inferenceModeForcesYoloAndDisablesCollectionExport() {
        Settings.applyOperatingMode(Settings.OperatingMode.Mode.INFERENCE)

        assertEquals(Settings.OperatingMode.Mode.INFERENCE, Settings.OperatingMode.current)
        assertEquals(Settings.DetectionMode.Mode.YOLO, Settings.DetectionMode.current)
        assertTrue(Settings.DetectionMode.enableYOLOinference)
        assertFalse(Settings.ExportData.frameIMG)
    }

    @Test
    fun dataCollectionModeOnlyEnablesCollectionSpecificControls() {
        val capabilities = Settings.capabilitiesFor(Settings.OperatingMode.Mode.DATA_COLLECTION)

        assertTrue(capabilities.traceCollectionSettingsEnabled)
        assertFalse(capabilities.inferenceTraceSettingsEnabled)
        assertFalse(capabilities.inferenceSettingsEnabled)
        assertFalse(capabilities.inferenceActionsEnabled)
    }

    @Test
    fun inferenceModeOnlyEnablesInferenceSpecificControls() {
        val capabilities = Settings.capabilitiesFor(Settings.OperatingMode.Mode.INFERENCE)

        assertFalse(capabilities.traceCollectionSettingsEnabled)
        assertTrue(capabilities.inferenceTraceSettingsEnabled)
        assertTrue(capabilities.inferenceSettingsEnabled)
        assertTrue(capabilities.inferenceActionsEnabled)
    }

    @Test
    fun collectionSupportsEveryAvailableTraceType() {
        val allTypes = Settings.Trace.Type.entries.toSet()

        Settings.Trace.updateCollectionTypes(allTypes)

        assertEquals(allTypes, Settings.Trace.collectionTypes)
        assertTrue(Settings.Trace.enableRAWtrace)
        assertTrue(Settings.Trace.enableSPLINEtrace)
    }

    @Test
    fun selectingOneFamilyOnlyUpdatesTheVisibleTraceFamily() {
        Settings.Trace.updateCollectionTypes(setOf(Settings.Trace.Type.RAW_CV))
        assertTrue(Settings.Trace.enableRAWtrace)
        assertFalse(Settings.Trace.enableSPLINEtrace)

        Settings.Trace.updateCollectionTypes(setOf(Settings.Trace.Type.SPLINE))
        assertFalse(Settings.Trace.enableRAWtrace)
        assertTrue(Settings.Trace.enableSPLINEtrace)
    }

    @Test
    fun inferenceTraceTypeCanBeChangedIndependentlyFromCollectionTypes() {
        Settings.Trace.updateCollectionTypes(setOf(Settings.Trace.Type.RAW_CV, Settings.Trace.Type.SPLINE_CV))
        Settings.Trace.updateInferenceType(Settings.Trace.Type.RAW)

        assertEquals(Settings.Trace.Type.RAW, Settings.Trace.inferenceType)
        assertEquals(
            setOf(Settings.Trace.Type.RAW_CV, Settings.Trace.Type.SPLINE_CV),
            Settings.Trace.collectionTypes
        )
    }

    @Test
    fun traceBoldnessControlsOverlayAndExportThickness() {
        Settings.Trace.updateBoldness(7)
        assertEquals(7, Settings.Trace.exportLineThickness)
        assertEquals(70, Settings.Trace.lineThickness)

        Settings.Trace.updateBoldness(0)
        assertEquals(Settings.Trace.MIN_BOLDNESS, Settings.Trace.boldness)
        Settings.Trace.updateBoldness(99)
        assertEquals(Settings.Trace.MAX_BOLDNESS, Settings.Trace.boldness)
    }

    @Test
    fun inferenceLabelSizeControlsFontScaleAndClampsToRange() {
        Settings.Inference.updateLabelSize(13)
        assertEquals(13, Settings.Inference.labelSize)
        assertEquals(1.3, Settings.Inference.labelFontScale, 0.0)

        Settings.Inference.updateLabelSize(0)
        assertEquals(Settings.Inference.MIN_LABEL_SIZE, Settings.Inference.labelSize)
        Settings.Inference.updateLabelSize(99)
        assertEquals(Settings.Inference.MAX_LABEL_SIZE, Settings.Inference.labelSize)
    }

    @Test
    fun collectionDirectoryUsesEmotionAndTraceType() {
        assertEquals(
            "Collected RoEmotion Data/User 01/Anxiety/Raw + CV Processing",
            collectionDirectory("User 01", "Anxiety", Settings.Trace.Type.RAW_CV)
        )
        assertEquals(
            "Collected RoEmotion Data/User 01/Anxiety/Kalman-Filter",
            collectionDirectory("User 01", "Anxiety", Settings.Trace.Type.SPLINE)
        )
        assertEquals(
            "Collected RoEmotion Data/User 01/Anxiety/Kalman Filter + CV Processing",
            collectionDirectory("User 01", "Anxiety", Settings.Trace.Type.SPLINE_CV)
        )
    }

    @Test
    fun traceFileNameContainsTimestampTypeAndSanitizedUser() {
        val capturedAt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
            .parse("2026-08-25 14:30:52.417")!!

        assertEquals(
            "2026_08_25_14_30_52_417_RAW_CV_User 01.jpg",
            traceFileName(capturedAt, Settings.Trace.Type.RAW_CV, "User 01")
        )
        assertEquals(
            "2026_08_25_14_30_52_417_SPLINE_CV_User _ 01.jpg",
            traceFileName(capturedAt, Settings.Trace.Type.SPLINE_CV, "User / 01")
        )
    }

    @Test
    fun collectionEmotionIsSanitizedAsOneFolderSegment() {
        assertEquals("Happy _ Excited", sanitizeCollectionPathSegment(" Happy / Excited "))
        assertEquals("Unspecified", sanitizeCollectionPathSegment("..."))
    }

    @Test
    fun detectionsAreRetainedUntilTheProcessingStateIsCleared() {
        val state = VideoProcessingState(userCount = 3)

        state.recordDetections(listOf(0))
        state.recordDetections(listOf(2))
        state.recordDetections(emptyList())

        assertEquals(listOf(0, 2), state.detectionOrderSnapshot())
        state.clearTraces()
        assertTrue(state.detectionOrderSnapshot().isEmpty())
    }
}
