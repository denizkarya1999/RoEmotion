package com.developer27.xemotion

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.developer27.xemotion.inference.EmotionModelCatalog
import com.developer27.xemotion.inference.PyTorchClassifier
import com.developer27.xemotion.inference.PyTorchModelLoader
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class EmotionModelsInstrumentedTest {
    @Test
    fun everyBundledModelRunsWithItsOwnInputAndLabelContract() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val expected = instrumentation.context.assets
            .open("emotion_model_smoke_predictions.json")
            .bufferedReader().use { JSONObject(it.readText()) }
        val loader = PyTorchModelLoader(instrumentation.targetContext)
        for (spec in EmotionModelCatalog.models) {
            val frames = List(spec.sequenceLength) {
                Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888).apply {
                    eraseColor(Color.BLACK)
                }
            }
            try {
                loader.loadEmotionClassifier(spec).use { classifier ->
                    val (label, probabilities) = classifier.classifySequence(frames)
                    assertTrue("${spec.id}: unexpected label $label", label in spec.labels)
                    assertEquals(spec.labels.size, probabilities.size)
                    assertTrue(probabilities.all { it.isFinite() && it in 0f..1f })
                    assertEquals("${spec.id}: probability total", 1f, probabilities.sum(), 0.001f)
                    if (!spec.isLegacy) {
                        val reference = expected.getJSONArray(spec.id)
                        probabilities.forEachIndexed { index, actual ->
                            assertEquals(
                                "${spec.id}: ${spec.labels[index]}",
                                reference.getDouble(index), actual.toDouble(), 0.001
                            )
                        }
                    }
                }
            } finally {
                frames.forEach(Bitmap::recycle)
            }
        }
    }

    @Test
    fun currentPreprocessingPreservesTheWholeTraceWithBlackPadding() {
        val source = Bitmap.createBitmap(224, 112, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.WHITE)
            setPixel(0, 0, Color.RED)
            setPixel(223, 111, Color.BLUE)
        }
        val padded = PyTorchClassifier.letterbox(source, 224, 224)
        try {
            assertEquals(Color.BLACK, padded.getPixel(112, 55))
            assertEquals(Color.BLACK, padded.getPixel(112, 168))
            assertEquals(Color.RED, padded.getPixel(0, 56))
            assertEquals(Color.BLUE, padded.getPixel(223, 167))
            assertEquals(Color.WHITE, padded.getPixel(112, 112))
            assertFalse(source.isRecycled)
        } finally {
            source.recycle()
            padded.recycle()
        }
    }
}
