package com.developer27.xemotion.inference

import org.opencv.core.Point
import java.util.ArrayDeque

/** Accumulates sparse YOLO detections until a renderable inference trace is available. */
internal class InferenceTraceBuffer(
    userCount: Int,
    private val capacity: Int
) {
    private val histories = Array(userCount) { ArrayDeque<Point>(capacity) }

    fun appendAndSnapshot(
        classId: Int,
        newPoints: List<Point>,
        minimumPoints: Int
    ): List<Point>? {
        if (classId !in histories.indices || newPoints.isEmpty()) return null

        val history = histories[classId]
        newPoints.forEach { point ->
            history.addLast(point)
            while (history.size > capacity) history.removeFirst()
        }
        return history.toList().takeIf { it.size >= minimumPoints }
    }

    fun clear() {
        histories.forEach(ArrayDeque<Point>::clear)
    }
}
