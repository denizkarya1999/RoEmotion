package com.developer27.xemotion

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.developer27.xemotion.inference.PyTorchModelLoader
import com.developer27.xemotion.videoprocessing.Settings
import com.developer27.xemotion.videoprocessing.VideoProcessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.runner.RunWith
import org.junit.Test
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.developer27.xemotion", appContext.packageName)
    }

    @Test
    fun openCvAndPyTorchShareThePackagedCxxRuntime() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertTrue("OpenCV native initialization failed", OpenCVLoader.initLocal())

        val matrix = Mat.eye(2, 2, CvType.CV_32F)
        try {
            assertEquals(1.0, matrix[0, 0][0], 0.0)
        } finally {
            matrix.release()
        }

        PyTorchModelLoader(appContext).loadEmotionClassifier().close()
    }

    @Test
    fun emotionClassifierProducesPredictionForFiveFrameSequence() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = PyTorchModelLoader(appContext).loadEmotionClassifier()
        val frames = List(5) { Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888) }

        try {
            val (label, probabilities) = classifier.classifySequence(frames)
            assertTrue("Emotion label was empty", label.isNotBlank())
            assertEquals(5, probabilities.size)
            assertEquals(1.0f, probabilities.sum(), 0.001f)
        } finally {
            frames.forEach(Bitmap::recycle)
            classifier.close()
        }
    }

    @Test
    fun contourTrackingProcessesAFrameWithoutCrashing() {
        Settings.applyOperatingMode(Settings.OperatingMode.Mode.DATA_COLLECTION)
        val processor = VideoProcessor()
        assertTrue("OpenCV was unavailable to VideoProcessor", processor.isOpenCvReady)

        val completed = CountDownLatch(1)
        var processed: Bitmap? = null
        val input = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        try {
            processor.processFrame(input) { result ->
                processed = result
                completed.countDown()
            }
            assertTrue("Contour frame processing timed out", completed.await(15, TimeUnit.SECONDS))
            assertNotNull("Contour frame processing failed", processed)
        } finally {
            processed?.recycle()
            // VideoProcessor owns and recycles the input bitmap after invoking the callback.
            processor.close()
        }
    }
}
