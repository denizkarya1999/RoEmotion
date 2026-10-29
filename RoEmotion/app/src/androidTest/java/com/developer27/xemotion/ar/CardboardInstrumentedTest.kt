package com.developer27.xemotion.ar

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.view.View
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CardboardInstrumentedTest {
    @Test fun failedStartDoesNotEnterImmersiveMode() {
        ActivityScenario.launch(CardboardTestActivity::class.java).use { scenario ->
            scenario.onActivity {
                it.allowStart = false
                val orientation = it.requestedOrientation
                it.manager.enterArMode(60000)
                assertFalse(it.manager.isArMode)
                assertEquals(orientation, it.requestedOrientation)
                assertEquals(View.GONE, it.binding.cardboardView.visibility)
            }
        }
    }

    @Test fun backExitsAndStopsOnlyAnArOwnedSession() {
        ActivityScenario.launch(CardboardTestActivity::class.java).use { scenario ->
            scenario.onActivity {
                val orientation = it.requestedOrientation
                it.manager.enterArMode(60000)
                assertTrue(it.manager.isArMode)
                assertEquals(View.GONE, it.binding.topControlsContainer.visibility)
                assertEquals(View.VISIBLE, it.binding.cardboardView.visibility)
                it.onBackPressedDispatcher.onBackPressed()
                assertFalse(it.manager.isArMode)
                assertFalse(it.processing)
                assertEquals(1, it.stops)
                assertEquals(orientation, it.requestedOrientation)
                assertEquals(View.VISIBLE, it.binding.topControlsContainer.visibility)
                it.processing = true
                it.manager.enterArMode(60000)
                it.binding.cardboardView.performLongClick()
                assertFalse(it.manager.isArMode)
                assertTrue(it.processing)
                assertEquals(1, it.stops)
            }
        }
    }

    @Test fun backgroundingCancelsSession() {
        ActivityScenario.launch(CardboardTestActivity::class.java).use { scenario ->
            scenario.onActivity { it.manager.enterArMode(60000) }
            scenario.moveToState(Lifecycle.State.CREATED)
            scenario.onActivity {
                assertFalse(it.manager.isArMode)
                assertEquals(1, it.stops)
            }
        }
    }

    @Test fun timerExitsAndRepeatedEntryDoesNotStartTwice() {
        ActivityScenario.launch(CardboardTestActivity::class.java).use { scenario ->
            scenario.onActivity {
                it.manager.enterArMode(100)
                it.manager.enterArMode(60000)
                assertEquals(1, it.starts)
            }
            Thread.sleep(300)
            scenario.onActivity {
                assertFalse(it.manager.isArMode)
                assertEquals(1, it.stops)
            }
        }
    }

    @Test fun bitmapAndStatusRenderIdenticallyInBothEyes() {
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val view = CardboardView(context)
            view.layout(0, 0, 1200, 600)
            val source = Bitmap.createBitmap(640, 360, Bitmap.Config.ARGB_8888)
            source.eraseColor(Color.GREEN)
            val output = Bitmap.createBitmap(1200, 600, Bitmap.Config.ARGB_8888)
            try {
                view.setFrame(source)
                view.draw(Canvas(output))
                // A live frame covers top/bottom and has no persistent timer or exit text.
                for (x in 2 until 598 step 7) for (y in 0 until 600 step 7) {
                    assertEquals(Color.GREEN, output.getPixel(x, y))
                    assertEquals(Color.GREEN, output.getPixel(x + 600, y))
                }
                // Include the outermost pixels and the seam: no margins or divider.
                for (x in listOf(0, 599, 600, 1199)) for (y in listOf(0, 300, 599)) {
                    assertEquals("Border at $x,$y", Color.GREEN, output.getPixel(x, y))
                }
                view.statusText = "Starting in 5"
                view.draw(Canvas(output))
                for (x in 2 until 598 step 7) for (y in 0 until 600 step 7) {
                    assertEquals("Eye mismatch at $x,$y", output.getPixel(x, y), output.getPixel(x + 600, y))
                }
                assertEquals(Color.GREEN, output.getPixel(300, 300))
                assertEquals(Color.GREEN, output.getPixel(10, 300))
                view.setFrame(null)
                source.recycle()
                view.draw(Canvas(output)) // no reference to a recycled bitmap
                assertEquals(Color.BLACK, output.getPixel(300, 350))
            } finally {
                if (!source.isRecycled) source.recycle()
                output.recycle()
            }
        }
    }
}
