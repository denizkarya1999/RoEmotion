package com.developer27.xemotion.inference

import org.opencv.core.Point
import java.util.ArrayDeque

/** Accumulates sparse YOLO detections until a renderable inference trace is available. */
internal class InferenceTraceBuffer(
    userCount: Int,
    private val capacity: Int
) {
    private val histories = Array(userCount) { ArrayDeque<Point>(capacity) }
    private val lastObservationMillis = LongArray(userCount) { NO_OBSERVATION }

    /** Returns true when this user's previous history expired before the new observation. */
    fun recordObservation(classId: Int, timestampMillis: Long, maxGapMillis: Long): Boolean {
        require(maxGapMillis >= 0L)
        if (classId !in histories.indices) return false
        val previous = lastObservationMillis[classId]
        val expired = previous != NO_OBSERVATION && timestampMillis - previous > maxGapMillis
        if (expired) clear(classId)
        lastObservationMillis[classId] = timestampMillis
        return expired
    }

    fun appendAndSnapshot(
        classId: Int,
        newPoints: List<Point>,
        minimumPoints: Int,
        transformPoint: (Point) -> Point = { it }
    ): List<Point>? {
        if (classId !in histories.indices || newPoints.isEmpty()) return null

        val history = histories[classId]
        newPoints.forEach { point ->
            history.addLast(transformPoint(point))
            while (history.size > capacity) history.removeFirst()
        }
        return history.toList().takeIf { it.size >= minimumPoints }
    }

    fun clear(classId: Int) {
        if (classId !in histories.indices) return
        histories[classId].clear()
        lastObservationMillis[classId] = NO_OBSERVATION
    }

    fun clear() {
        histories.indices.forEach(::clear)
    }

    private companion object {
        const val NO_OBSERVATION = -1L
    }
}
