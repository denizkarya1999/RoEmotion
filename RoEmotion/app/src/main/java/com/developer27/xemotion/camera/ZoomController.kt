package com.developer27.xemotion.camera

import android.os.Handler
import android.view.MotionEvent
import android.view.View

/** Owns press-and-hold zoom gestures independently from camera session code. */
class ZoomController(
    private val zoomInView: View,
    private val zoomOutView: View,
    private val handler: Handler,
    private val onZoomChanged: (Float) -> Unit
) {
    private var zoomLevel = MIN_ZOOM
    private var activeRunnable: Runnable? = null

    fun bind() {
        zoomInView.setOnTouchListener(zoomTouchListener(ZOOM_STEP))
        zoomOutView.setOnTouchListener(zoomTouchListener(-ZOOM_STEP))
    }

    fun close() {
        activeRunnable?.let(handler::removeCallbacks)
        activeRunnable = null
        zoomInView.setOnTouchListener(null)
        zoomOutView.setOnTouchListener(null)
    }

    private fun zoomTouchListener(delta: Float) = View.OnTouchListener { view, event ->
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                startRepeating(delta)
                true
            }
            MotionEvent.ACTION_UP -> {
                stopRepeating()
                view.performClick()
                true
            }
            MotionEvent.ACTION_CANCEL -> {
                stopRepeating()
                true
            }
            else -> false
        }
    }

    private fun startRepeating(delta: Float) {
        stopRepeating()
        activeRunnable = object : Runnable {
            override fun run() {
                zoomLevel = (zoomLevel + delta).coerceIn(MIN_ZOOM, MAX_ZOOM)
                onZoomChanged(zoomLevel)
                handler.postDelayed(this, REPEAT_DELAY_MILLIS)
            }
        }.also(handler::post)
    }

    private fun stopRepeating() {
        activeRunnable?.let(handler::removeCallbacks)
        activeRunnable = null
    }

    private companion object {
        const val MIN_ZOOM = 1f
        const val MAX_ZOOM = 10f
        const val ZOOM_STEP = 0.1f
        const val REPEAT_DELAY_MILLIS = 50L
    }
}
