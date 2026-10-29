package com.developer27.xemotion.ar

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.drawing.BoundingBox
import com.developer27.xemotion.videoprocessing.drawing.OverlayStyle
import com.developer27.xemotion.videoprocessing.drawing.UserEmotionDisplay
import com.developer27.xemotion.videoprocessing.drawing.VideoDrawingHelper
import org.junit.Assert.*
import org.junit.BeforeClass
import org.junit.Before
import org.junit.After
import com.developer27.xemotion.videoprocessing.AppearanceSettings
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.Point
import com.developer27.xemotion.videoprocessing.drawing.traces.TraceRenderer
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar

@RunWith(AndroidJUnit4::class)
class ArOverlayInstrumentedTest {
    private lateinit var originalProfiles: Map<AppearanceSettings.Mode, AppearanceSettings.Profile>
    @Before fun useDefaultProfiles() {
        originalProfiles = AppearanceSettings.Mode.entries.associateWith(AppearanceSettings::profile)
        for (mode in AppearanceSettings.Mode.entries) for (field in AppearanceSettings.Field.entries) {
            val defaults = if (mode == AppearanceSettings.Mode.AR) AppearanceSettings.DEFAULT_AR else AppearanceSettings.DEFAULT_CV
            AppearanceSettings.update(mode, field, defaults.value(field))
        }
    }
    @After fun restoreProfiles() {
        originalProfiles.forEach { (mode, profile) ->
            AppearanceSettings.Field.entries.forEach { AppearanceSettings.update(mode, it, profile.value(it)) }
        }
    }

    private val white = Scalar(255.0, 255.0, 255.0)
    private val box = BoundingBox(300f, 400f, 600f, 650f, 0.95f, 0)
    private val drawing = VideoDrawingHelper { UserEmotionDisplay("Joy", white, 95f) }

    @Test fun arLabelsOccupyLessSpaceThanHandheldLabels() {
        val normal = Mat.zeros(768, 1024, CvType.CV_8UC3)
        val ar = Mat.zeros(768, 1024, CvType.CV_8UC3)
        try {
            drawing.drawTopLabel(normal, box, "User 1 | Joy 95%", white)
            drawing.drawTopLabel(ar, box, "User 1 | Joy 95%", white, OverlayStyle.AR)
            assertTrue(ink(ar) > 0)
            assertTrue("AR text should be substantially smaller", ink(ar) < ink(normal) / 2)
        } finally {
            normal.release()
            ar.release()
        }
    }

    @Test fun arOutlinesAreThinnerWithoutMovingDetectionBoundsOrChangingSettings() {
        val normal = Mat.zeros(768, 1024, CvType.CV_8UC3)
        val ar = Mat.zeros(768, 1024, CvType.CV_8UC3)
        val originalThickness = Settings.BoundingBox.boxThickness
        val originalFont = Settings.Inference.labelFontScale
        try {
            drawing.drawUserBoxAndLabel(normal, box, 0)
            drawing.drawUserBoxAndLabel(ar, box, 0, OverlayStyle.AR)
            assertTrue(rowInk(ar, 500) < rowInk(normal, 500))
            for (x in listOf(300, 600)) {
                assertTrue(normal.get(500, x).any { it > 0 })
                assertTrue(ar.get(500, x).any { it > 0 })
            }
            val normalLabels = normal.rowRange(0, 390)
            val arLabels = ar.rowRange(0, 390)
            try {
                assertTrue(ink(arLabels) > 0)
                assertTrue(ink(arLabels) < ink(normalLabels) / 2)
            } finally {
                normalLabels.release()
                arLabels.release()
            }
            assertEquals(originalThickness, Settings.BoundingBox.boxThickness)
            assertEquals(originalFont, Settings.Inference.labelFontScale, 0.0)
        } finally {
            normal.release()
            ar.release()
        }
    }

    @Test fun traceStrokesUseEachModesIndependentSelectedWidth() {
        AppearanceSettings.update(AppearanceSettings.Mode.CV, AppearanceSettings.Field.TRACE, 3)
        AppearanceSettings.update(AppearanceSettings.Mode.AR, AppearanceSettings.Field.TRACE, 9)
        val cv = Mat.zeros(768, 1024, CvType.CV_8UC3)
        val ar = Mat.zeros(768, 1024, CvType.CV_8UC3)
        try {
            val path = listOf(Point(350.0, 500.0), Point(550.0, 500.0))
            TraceRenderer.drawPathInBox(cv, box, path, type = Settings.Trace.Type.RAW,
                color = white, thickness = OverlayStyle.HANDHELD.traceThickness)
            TraceRenderer.drawPathInBox(ar, box, path, type = Settings.Trace.Type.RAW,
                color = white, thickness = OverlayStyle.AR.traceThickness)
            val cvColumn = cv.col(450)
            val arColumn = ar.col(450)
            try {
                assertTrue(ink(cvColumn) > 0)
                assertTrue(ink(arColumn) > ink(cvColumn))
            } finally {
                cvColumn.release()
                arColumn.release()
            }
        } finally {
            cv.release()
            ar.release()
        }
    }

    private fun ink(mat: Mat): Int {
        val flat = mat.reshape(1)
        return try { Core.countNonZero(flat) } finally { flat.release() }
    }

    private fun rowInk(mat: Mat, y: Int): Int {
        val row = mat.row(y)
        return try { ink(row) } finally { row.release() }
    }

    companion object {
        @BeforeClass @JvmStatic fun loadOpenCv() { assertTrue(OpenCVLoader.initLocal()) }
    }
}
