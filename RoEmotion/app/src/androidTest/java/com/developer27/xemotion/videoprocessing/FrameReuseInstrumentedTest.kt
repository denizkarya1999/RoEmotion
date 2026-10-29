package com.developer27.xemotion.videoprocessing

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class FrameReuseInstrumentedTest {
    @Test fun mutableYoloFrameKeepsPixelsAndTransfersOwnershipToDisplay() =
        checkFrame(Settings.DetectionMode.Mode.YOLO, mutable = true)

    @Test fun mutableContourFrameTransfersOwnershipToDisplay() =
        checkFrame(Settings.DetectionMode.Mode.CONTOUR, mutable = true)

    @Test fun immutableInputStillProducesIndependentOutput() =
        checkFrame(Settings.DetectionMode.Mode.YOLO, mutable = false)

    private fun checkFrame(mode: Settings.DetectionMode.Mode, mutable: Boolean) {
        val previousMode = Settings.DetectionMode.current
        val processor = VideoProcessor()
        val source = Bitmap.createBitmap(640, 360, Bitmap.Config.ARGB_8888)
        source.eraseColor(if (mode == Settings.DetectionMode.Mode.YOLO) Color.GREEN else Color.BLACK)
        val input = if (mutable) source else source.copy(Bitmap.Config.ARGB_8888, false).also { source.recycle() }
        var output: Bitmap? = null
        try {
            Settings.DetectionMode.current = mode
            assertTrue(processor.isOpenCvReady)
            val completed = CountDownLatch(1)
            processor.processFrame(input, arMode = true) { result ->
                output = result
                completed.countDown()
            }
            assertTrue(completed.await(10, TimeUnit.SECONDS))
            // Let the asynchronous processor's cleanup run after the callback.
            InstrumentationRegistry.getInstrumentation().waitForIdleSync()
            Thread.sleep(100)
            val result = requireNotNull(output)
            assertFalse("Displayed output was recycled by the processor", result.isRecycled)
            assertEquals(640, result.width)
            assertEquals(360, result.height)
            if (mutable) assertSame("Preview processing allocated an extra bitmap", input, result)
            else {
                assertNotSame(input, result)
                assertTrue("Unneeded input should be released", input.isRecycled)
            }
            assertEquals(if (mode == Settings.DetectionMode.Mode.YOLO) Color.GREEN else Color.BLACK,
                result.getPixel(320, 180))
        } finally {
            processor.close()
            output?.takeUnless(Bitmap::isRecycled)?.recycle()
            if (!input.isRecycled) input.recycle()
            Settings.DetectionMode.current = previousMode
        }
    }
}
