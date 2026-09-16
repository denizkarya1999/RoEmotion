package com.developer27.xemotion.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
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

    @Test
    fun gapOverTrainingLimitClearsOnlyTheReturningUsersHistory() {
        val buffer = InferenceTraceBuffer(userCount = 2, capacity = 10)
        buffer.recordObservation(0, 1_000L, 10_000L)
        buffer.recordObservation(1, 1_000L, 10_000L)
        buffer.appendAndSnapshot(0, listOf(Point(1.0, 1.0)), minimumPoints = 2)
        buffer.appendAndSnapshot(1, listOf(Point(10.0, 10.0)), minimumPoints = 2)

        assertTrue(buffer.recordObservation(0, 11_001L, 10_000L))
        assertNull(buffer.appendAndSnapshot(0, listOf(Point(2.0, 2.0)), minimumPoints = 2))
        assertEquals(
            listOf(Point(10.0, 10.0), Point(20.0, 20.0)),
            buffer.appendAndSnapshot(1, listOf(Point(20.0, 20.0)), minimumPoints = 2)
        )
    }

    @Test
    fun exactMaximumGapRetainsHistoryAndLaterObservationsRefreshTheClock() {
        val buffer = InferenceTraceBuffer(userCount = 1, capacity = 10)
        assertFalse(buffer.recordObservation(0, 0L, 10_000L))
        buffer.appendAndSnapshot(0, listOf(Point(1.0, 1.0)), minimumPoints = 2)

        assertFalse(buffer.recordObservation(0, 10_000L, 10_000L))
        assertFalse(buffer.recordObservation(0, 19_000L, 10_000L))
        assertEquals(
            listOf(Point(1.0, 1.0), Point(2.0, 2.0)),
            buffer.appendAndSnapshot(0, listOf(Point(2.0, 2.0)), minimumPoints = 2)
        )
    }

    @Test
    fun clearingSessionAlsoForgetsThePreviousObservationTime() {
        val buffer = InferenceTraceBuffer(userCount = 1, capacity = 10)
        buffer.recordObservation(0, 1_000L, 10_000L)
        buffer.clear()

        assertFalse(buffer.recordObservation(0, 30_000L, 10_000L))
    }

    @Test
    fun smoothingIsAppliedOnceToNewPointsInsteadOfAgainToRetainedHistory() {
        val buffer = InferenceTraceBuffer(userCount = 1, capacity = 10)
        var calls = 0
        val smooth: (Point) -> Point = { point ->
            calls++
            Point(point.x + 100.0, point.y + 100.0)
        }
        assertNull(buffer.appendAndSnapshot(0, listOf(Point(1.0, 1.0)), 2, smooth))
        val result = buffer.appendAndSnapshot(0, listOf(Point(2.0, 2.0)), 2, smooth)

        assertEquals(2, calls)
        assertEquals(listOf(Point(101.0, 101.0), Point(102.0, 102.0)), result)
    }

}
