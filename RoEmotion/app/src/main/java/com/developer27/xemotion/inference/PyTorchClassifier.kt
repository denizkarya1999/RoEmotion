package com.developer27.xemotion.inference

import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean

/** Runs emotion classification for a prepared sequence of trace images. */
class PyTorchClassifier internal constructor(
    private val module: Module,
    private val inputWidth: Int = 224,
    private val inputHeight: Int = 224,
    private val mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
    private val std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)
) : Closeable {
    private val closed = AtomicBoolean(false)

    fun classifySequence(frames: List<Bitmap>): Pair<String, FloatArray> {
        check(!closed.get()) { "The emotion classifier is closed." }
        require(frames.isNotEmpty()) { "At least one frame is required for classification." }

        val valuesPerFrame = 3 * inputHeight * inputWidth
        val sequenceValues = FloatArray(valuesPerFrame * frames.size)

        frames.forEachIndexed { index, bitmap ->
            val processed = preprocess(bitmap)
            val values = normalizedRgbChannels(processed)
            System.arraycopy(values, 0, sequenceValues, index * valuesPerFrame, valuesPerFrame)
            if (processed !== bitmap) processed.recycle()
        }

        val sequenceTensor = Tensor.fromBlob(
            sequenceValues,
            longArrayOf(1, frames.size.toLong(), 3, inputHeight.toLong(), inputWidth.toLong())
        )
        val logits = module.forward(IValue.from(sequenceTensor)).toTensor().dataAsFloatArray
        val probabilities = softmax(logits)
        val bestIndex = probabilities.indices.maxByOrNull(probabilities::get) ?: 0
        return emotionLabels.getOrElse(bestIndex) { "Unknown" } to probabilities
    }

    fun logProbabilities(probabilities: FloatArray) {
        probabilities.forEachIndexed { index, probability ->
            Log.d(TAG, "${emotionLabels.getOrNull(index) ?: "Label$index"}: $probability")
        }
    }

    private fun preprocess(bitmap: Bitmap): Bitmap {
        val resized = resizeShortSide(bitmap, RESIZE_SHORT_SIDE)
        val cropped = centerCrop(resized, inputWidth, inputHeight)
        if (resized !== bitmap && resized !== cropped) resized.recycle()
        return cropped
    }

    private fun resizeShortSide(bitmap: Bitmap, shortSide: Int): Bitmap {
        if (bitmap.width <= 0 || bitmap.height <= 0) return bitmap
        val scale = if (bitmap.width < bitmap.height) {
            shortSide.toFloat() / bitmap.width
        } else {
            shortSide.toFloat() / bitmap.height
        }
        val width = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val height = (bitmap.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    private fun centerCrop(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        val cropWidth = width.coerceAtMost(bitmap.width)
        val cropHeight = height.coerceAtMost(bitmap.height)
        val x = ((bitmap.width - cropWidth) / 2).coerceAtLeast(0)
        val y = ((bitmap.height - cropHeight) / 2).coerceAtLeast(0)
        return Bitmap.createBitmap(bitmap, x, y, cropWidth, cropHeight)
    }

    /** Produces the same normalized channel-first RGB layout expected by the ResNet model. */
    private fun normalizedRgbChannels(bitmap: Bitmap): FloatArray {
        val pixelCount = bitmap.width * bitmap.height
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val values = FloatArray(pixelCount * 3)
        pixels.forEachIndexed { index, pixel ->
            values[index] = (((pixel shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            values[pixelCount + index] = (((pixel shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            values[2 * pixelCount + index] = ((pixel and 0xFF) / 255f - mean[2]) / std[2]
        }
        return values
    }

    private fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return floatArrayOf()
        val maxLogit = logits.maxOrNull() ?: 0f
        val exponentials = logits.map { Math.exp((it - maxLogit).toDouble()).toFloat() }
        val total = exponentials.sum()
        if (total == 0f || !total.isFinite()) return FloatArray(logits.size)
        return exponentials.map { it / total }.toFloatArray()
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) module.destroy()
    }

    companion object {
        val emotionLabels = listOf("Angry", "Anxiety", "Disgust", "Excitement", "Sadness")

        private const val TAG = "PyTorchClassifier"
        private const val RESIZE_SHORT_SIDE = 256
    }
}
