package com.developer27.xemotion.videoprocessing

import org.opencv.core.Point
import org.opencv.core.Scalar
import com.developer27.xemotion.videoprocessing.drawing.UserEmotionDisplay
import java.util.LinkedList

/** Mutable state owned by one [VideoProcessor] instance. */
internal class VideoProcessingState(private val userCount: Int = 3) {
    val rawTrace = LinkedList<Point>()
    val smoothTrace = LinkedList<Point>()
    private val perUserRawTrace = Array(userCount) { LinkedList<Point>() }
    private val emotions = Array(userCount) { "" }
    private val emotionColors = Array(userCount) { DEFAULT_COLOR }
    private val emotionConfidence = FloatArray(userCount) { Float.NaN }
    private val emotionLock = Any()

    private val detectionOrder = LinkedList<Int>()

    fun detectionOrderSnapshot(): List<Int> = synchronized(detectionOrder) {
        detectionOrder.toList()
    }

    fun recordDetections(classIds: List<Int>) = synchronized(detectionOrder) {
        classIds.forEach { classId ->
            if (classId in 0 until userCount && classId !in detectionOrder) {
                detectionOrder.add(classId)
            }
        }
    }

    fun appendPerUserTrace(classId: Int, point: Point, capacity: Int): List<Point> {
        val trace = perUserRawTrace[classId.coerceIn(0, userCount - 1)]
        return synchronized(trace) {
            trace.add(point)
            while (trace.size > capacity) trace.removeFirst()
            trace.toList()
        }
    }

    fun perUserTraceSnapshot(classId: Int): List<Point> {
        val trace = perUserRawTrace[classId.coerceIn(0, userCount - 1)]
        return synchronized(trace) { trace.toList() }
    }

    fun drainPerUserTrace(classId: Int): List<Point> {
        val trace = perUserRawTrace[classId.coerceIn(0, userCount - 1)]
        return synchronized(trace) {
            trace.toList().also { trace.clear() }
        }
    }

    fun smoothTraceSnapshot(): List<Point> = synchronized(rawTrace) {
        smoothTrace.toList()
    }

    fun rawTraceSnapshot(): List<Point> = synchronized(rawTrace) {
        rawTrace.toList()
    }

    fun setEmotion(classId: Int, emotion: String, color: Scalar?, confidence: Float?) {
        val id = classId.coerceIn(0, userCount - 1)
        synchronized(emotionLock) {
            emotions[id] = emotion
            emotionColors[id] = color ?: DEFAULT_COLOR
            confidence?.let { emotionConfidence[id] = it }
        }
    }

    fun emotionDisplaySnapshot(classId: Int): UserEmotionDisplay {
        val id = classId.coerceIn(0, userCount - 1)
        return synchronized(emotionLock) {
            UserEmotionDisplay(emotions[id], emotionColors[id], emotionConfidence[id])
        }
    }

    fun clearTraces() {
        synchronized(rawTrace) {
            rawTrace.clear()
            smoothTrace.clear()
        }
        perUserRawTrace.forEach { trace -> synchronized(trace) { trace.clear() } }
        synchronized(detectionOrder) { detectionOrder.clear() }
    }

    fun clearEmotions() {
        synchronized(emotionLock) {
            emotions.indices.forEach { index ->
                emotions[index] = ""
                emotionColors[index] = DEFAULT_COLOR
                emotionConfidence[index] = Float.NaN
            }
        }
    }

    private companion object {
        val DEFAULT_COLOR = Scalar(255.0, 255.0, 255.0)
    }
}
