package com.developer27.xemotion.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import org.opencv.core.Point

class InferenceTraceBufferTest {
    @Test
    fun sparseDetectionTicksAccumulateUntilTraceIsRenderable() {
        val buffer = InferenceTraceBuffer(userCount = 3, capacity = 10)

        assertNull(buffer.appendAndSnapshot(1, listOf(Point(1.0, 1.0)), minimumPoints = 3))
        assertNull(buffer.appendAndSnapshot(1, listOf(Point(2.0, 2.0)), minimumPoints = 3))

        val trace = buffer.appendAndSnapshot(1, listOf(Point(3.0, 3.0)), minimumPoints = 3)
        assertEquals(3, trace?.size)
        assertEquals(Point(1.0, 1.0), trace?.first())
        assertEquals(Point(3.0, 3.0), trace?.last())
    }

    @Test
    fun historyUsesBoundedRollingCapacity() {
        val buffer = InferenceTraceBuffer(userCount = 1, capacity = 3)

        val trace = buffer.appendAndSnapshot(
            classId = 0,
            newPoints = (1..5).map { value -> Point(value.toDouble(), value.toDouble()) },
            minimumPoints = 2
        )

        assertEquals(listOf(Point(3.0, 3.0), Point(4.0, 4.0), Point(5.0, 5.0)), trace)
    }

    @Test
    fun clearRemovesPendingPointsBetweenSessions() {
        val buffer = InferenceTraceBuffer(userCount = 1, capacity = 10)
        buffer.appendAndSnapshot(0, listOf(Point(1.0, 1.0)), minimumPoints = 2)

        buffer.clear()

        assertNull(buffer.appendAndSnapshot(0, listOf(Point(2.0, 2.0)), minimumPoints = 2))
    }
}
