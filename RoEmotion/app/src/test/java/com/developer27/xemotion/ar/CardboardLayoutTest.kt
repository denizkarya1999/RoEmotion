package com.developer27.xemotion.ar

import org.junit.Assert.*
import org.junit.Test

class CardboardLayoutTest {
    @Test fun eyesFillTheirViewportsWithTheSameAspectPreservingCrop() {
        for ((width, height) in listOf(2400 to 1080, 1921 to 1080, 1080 to 2400)) {
            for ((sourceWidth, sourceHeight) in listOf(1280 to 720, 720 to 1280)) {
                val left = CardboardLayout.eyeFrame(width, height, sourceWidth, sourceHeight, false)
                val right = CardboardLayout.eyeFrame(width, height, sourceWidth, sourceHeight, true)
                assertEquals(left.width, right.width, 0f)
                assertEquals(left.height, right.height, 0f)
                assertEquals(left.top, right.top, 0f)
                assertEquals(width / 2f, right.left - left.left, 0.001f)
                assertEquals(sourceWidth.toFloat() / sourceHeight, left.width / left.height, 0.001f)
                assertTrue(left.left <= 0.001f && left.top <= 0.001f)
                assertTrue(left.left + left.width >= width / 2f - 0.001f)
                assertTrue(right.left + right.width >= width - 0.001f)
                assertTrue(left.top + left.height >= height - 0.001f)
                assertEquals(width / 4f, left.left + left.width / 2f, 0.001f)
                assertEquals(height / 2f, left.top + left.height / 2f, 0.001f)
            }
        }
    }

    @Test fun landscapeCameraFillsFullHeightAndCropsSidesSymmetrically() {
        val frame = CardboardLayout.eyeFrame(2400, 1080, 1280, 720, false)
        assertEquals(1080f, frame.height, 0.001f)
        assertEquals(0f, frame.top, 0.001f)
        assertTrue(frame.width > 1200f)
        assertEquals(-frame.left, frame.left + frame.width - 1200f, 0.001f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun zeroSourceIsRejected() { CardboardLayout.eyeFrame(2400, 1080, 0, 720, false) }

    @Test(expected = IllegalArgumentException::class)
    fun zeroViewportIsRejected() { CardboardLayout.eyeFrame(0, 1080, 1280, 720, false) }
}
