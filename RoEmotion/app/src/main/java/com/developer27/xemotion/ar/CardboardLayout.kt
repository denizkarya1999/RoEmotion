package com.developer27.xemotion.ar

/** Pure geometry: fill each eye with the same centered camera crop, without stretching. */
object CardboardLayout {
    data class Frame(val left: Float, val top: Float, val width: Float, val height: Float)

    fun eyeFrame(width: Int, height: Int, sourceWidth: Int, sourceHeight: Int,
                 rightEye: Boolean): Frame {
        require(width > 0 && height > 0 && sourceWidth > 0 && sourceHeight > 0)
        val eyeWidth = width / 2f
        val scale = maxOf(eyeWidth / sourceWidth, height.toFloat() / sourceHeight)
        val frameWidth = sourceWidth * scale
        val frameHeight = sourceHeight * scale
        return Frame((if (rightEye) eyeWidth else 0f) + (eyeWidth - frameWidth) / 2f,
            (height - frameHeight) / 2f, frameWidth, frameHeight)
    }
}
