package com.developer27.xemotion.videoprocessing

import com.developer27.xemotion.videoprocessing.AppearanceSettings.Field
import com.developer27.xemotion.videoprocessing.AppearanceSettings.Mode
import com.developer27.xemotion.videoprocessing.drawing.OverlayStyle
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class AppearanceSettingsTest {
    @Before @After fun defaults() {
        for (mode in Mode.entries) for (field in Field.entries) {
            val profile = if (mode == Mode.AR) AppearanceSettings.DEFAULT_AR else AppearanceSettings.DEFAULT_CV
            AppearanceSettings.update(mode, field, profile.value(field))
        }
    }

    @Test fun arDefaultsAreSmallerForEveryControl() {
        for (field in Field.entries) {
            assertTrue(AppearanceSettings.profile(Mode.AR).value(field) < AppearanceSettings.profile(Mode.CV).value(field))
        }
    }

    @Test fun cvEditsDoNotChangeArOrExportWidths() {
        val ar = AppearanceSettings.profile(Mode.AR)
        val export = Settings.Trace.exportLineThickness
        for (field in Field.entries) AppearanceSettings.update(Mode.CV, field, 19)
        assertEquals(ar, AppearanceSettings.profile(Mode.AR))
        assertEquals(export, Settings.Trace.exportLineThickness)
        assertEquals(19, OverlayStyle.HANDHELD.boxThickness)
        assertEquals(19, OverlayStyle.HANDHELD.traceThickness)
        assertEquals(1.9, OverlayStyle.HANDHELD.fontScale, 0.0)
    }

    @Test fun arEditsDoNotChangeCvAndAreNotScaledFromCv() {
        val cv = AppearanceSettings.profile(Mode.CV)
        AppearanceSettings.update(Mode.AR, Field.BOX, 7)
        AppearanceSettings.update(Mode.AR, Field.TEXT, 12)
        AppearanceSettings.update(Mode.AR, Field.TRACE, 4)
        val ar = OverlayStyle.AR
        assertEquals(cv, AppearanceSettings.profile(Mode.CV))
        assertEquals(7, ar.boxThickness)
        assertEquals(1.2, ar.fontScale, 0.0)
        assertEquals(4, ar.traceThickness)
    }

    @Test fun capturedFrameStyleCannotChangeMidFrame() {
        val captured = OverlayStyle.AR
        AppearanceSettings.update(Mode.AR, Field.TRACE, 18)
        assertEquals(2, captured.traceThickness)
        assertEquals(18, OverlayStyle.AR.traceThickness)
    }

    @Test fun everyControlIsClampedIndependently() {
        for (mode in Mode.entries) for (field in Field.entries) {
            AppearanceSettings.update(mode, field, -100)
            assertEquals(field.minimum, AppearanceSettings.profile(mode).value(field))
            AppearanceSettings.update(mode, field, 999)
            assertEquals(field.maximum, AppearanceSettings.profile(mode).value(field))
        }
    }
}
