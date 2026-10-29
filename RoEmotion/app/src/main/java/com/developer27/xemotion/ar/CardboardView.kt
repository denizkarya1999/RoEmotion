package com.developer27.xemotion.ar

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.os.SystemClock
import android.view.View
import com.developer27.xemotion.R

/**
 * A binocular monitor for a single rear camera, not synthetic stereo depth.
 * References the one processed frame; never copies it or runs inference per eye.
 * The Activity owns bitmap lifetime and must clear this view before recycling.
 */
class CardboardView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val framePaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textAlign = Paint.Align.CENTER
    }
    private val destination = RectF()
    private var frame: Bitmap? = null
    private var frameTime = 0L
    private val expireFrame = Runnable { invalidate() }
    var statusText: String = ""
        set(value) { field = value; invalidate() }

    init {
        setBackgroundColor(Color.BLACK)
        isClickable = true
        isLongClickable = true
        contentDescription = context.getString(R.string.cardboard_exit_hint)
        // A held Cardboard touch trigger (or long press after removal) exits.
        // Ordinary taps do nothing, so inserting the phone cannot end the session.
    }

    fun setFrame(bitmap: Bitmap?) {
        frame = bitmap
        frameTime = SystemClock.uptimeMillis()
        removeCallbacks(expireFrame)
        if (bitmap != null) postDelayed(expireFrame, STALE_FRAME_MILLIS)
        invalidate()
    }

    override fun onDetachedFromWindow() {
        removeCallbacks(expireFrame)
        super.onDetachedFromWindow()
    }

    private companion object {
        const val STALE_FRAME_MILLIS = 2_000L
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return
        val bitmap = frame?.takeUnless {
            it.isRecycled || SystemClock.uptimeMillis() - frameTime >= STALE_FRAME_MILLIS
        }
        val eyeWidth = width / 2f
        for (eye in 0..1) {
            val checkpoint = canvas.save()
            canvas.clipRect(eye * eyeWidth, 0f, (eye + 1) * eyeWidth, height.toFloat())
            if (bitmap != null) {
                val bounds = CardboardLayout.eyeFrame(width, height, bitmap.width, bitmap.height,
                    eye == 1)
                destination.set(bounds.left, bounds.top, bounds.left + bounds.width,
                    bounds.top + bounds.height)
                canvas.drawBitmap(bitmap, null, destination, framePaint)
            }
            textPaint.textSize = minOf(eyeWidth * 0.045f, height * 0.065f)
            val centerX = (eye + 0.5f) * eyeWidth
            canvas.drawText(statusText, centerX, height * 0.22f, textPaint)
            if (bitmap == null) {
                canvas.drawText(context.getString(R.string.cardboard_waiting), centerX,
                    height * 0.5f, textPaint)
            }
            // Keep normal viewing free of HUD text; instructions remain available while waiting.
            if (bitmap == null) {
                textPaint.textSize *= 0.7f
                canvas.drawText(context.getString(R.string.cardboard_exit_hint), centerX,
                    height * 0.84f, textPaint)
            }
            canvas.restoreToCount(checkpoint)
        }
    }
}
