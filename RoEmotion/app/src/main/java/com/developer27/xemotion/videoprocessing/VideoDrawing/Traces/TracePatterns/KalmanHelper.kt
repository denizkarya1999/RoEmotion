package com.developer27.xemotion.videoprocessing.VideoDrawing.Traces.TracePatterns

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.video.KalmanFilter

// --------------------------------------------------
// KalmanHelper
// --------------------------------------------------
object KalmanBank {
    // One filter per class (0=User_1, 1=User_2, 2=User_3)
    private val filters: Array<KalmanFilter?> = arrayOfNulls(3)

    private fun newKF(): KalmanFilter {
        val kf = KalmanFilter(4, 2)
        kf._transitionMatrix = Mat.eye(4, 4, CvType.CV_32F).apply {
            put(0, 2, 1.0)
            put(1, 3, 1.0)
        }
        kf._measurementMatrix = Mat.eye(2, 4, CvType.CV_32F)
        kf._processNoiseCov   = Mat.eye(4, 4, CvType.CV_32F).apply { setTo(Scalar(1e-4)) }
        kf._measurementNoiseCov = Mat.eye(2, 2, CvType.CV_32F).apply { setTo(Scalar(1e-2)) }
        kf._errorCovPost = Mat.eye(4, 4, CvType.CV_32F)
        return kf
    }

    /** Reset all filters (call from VideoProcessor.reset()) */
    fun initKalmanFilter() {
        for (i in filters.indices) filters[i] = null
    }

    /**
     * Correct position for a given classId (0,1,2).
     * If you don’t care about class, just pass 0.
     */
    fun applyKalmanFilter(point: Point, classId: Int = 0): Pair<Double, Double> {
        val id = classId.coerceIn(0, filters.lastIndex)
        val kf = filters[id] ?: newKF().also { filters[id] = it }

        val measurement = Mat(2, 1, CvType.CV_32F).apply {
            put(0, 0, point.x)
            put(1, 0, point.y)
        }
        kf.predict()
        val corrected = kf.correct(measurement)

        val fx = corrected[0, 0][0]
        val fy = corrected[1, 0][0]
        return fx to fy
    }

    fun reset() {
        for (i in filters.indices) filters[i] = null
    }

    /** Per-class correction; use classId in [0,2]. */
    fun correct(classId: Int, p: Point): Point {
        val id = classId.coerceIn(0, 2)
        val kf = filters[id] ?: newKF().also { filters[id] = it }
        val z = Mat(2, 1, CvType.CV_32F).apply {
            put(0, 0, p.x); put(1, 0, p.y)
        }
        kf.predict()
        val corr = kf.correct(z)
        return Point(corr[0,0][0], corr[1,0][0])
    }
}